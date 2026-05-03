"""
Generate synthetic driving preference dataset for DPO training.
Creates (prompt, chosen, rejected) pairs covering key driving safety scenarios.
Output: dataset/driving_preferences.jsonl
"""
import json
import os
import random

random.seed(42)

SYSTEM = "You are DriveMind-V, an autonomous driving assistant. Analyze the driving scenario and recommend the safest action."

# ── Scenario templates ──────────────────────────────────────────────────────

SCENARIOS = [
    # 1. Pedestrian interactions
    {
        "category": "pedestrian",
        "prompts": [
            "A pedestrian is crossing at a crosswalk ahead. The light just turned green for you. What should you do?",
            "A child suddenly runs into the street between parked cars. What is the correct response?",
            "A pedestrian is on the sidewalk and looks like they might step into the road. How do you proceed?",
            "You approach a crosswalk with a pedestrian waiting to cross. The light is green for you. What do you do?",
            "A group of pedestrians is crossing slowly on a marked crosswalk. What is the best action?",
            "An elderly person is still crossing as the walk signal starts flashing. What do you do?",
            "A pedestrian is jaywalking midblock ahead of you. What should you do?",
        ],
        "chosen": [
            "Yield to the pedestrian and wait until they have fully cleared the crosswalk before proceeding, even though the light is green.",
            "Immediately brake hard, reducing speed to zero if necessary, and wait for the child to reach safety before continuing.",
            "Slow down, cover the brake, and monitor the pedestrian closely. Be prepared to stop if they step into the road.",
            "Slow down and yield. Pedestrians have the right of way at crosswalks regardless of the signal state.",
            "Slow down and wait patiently for all pedestrians to fully cross before proceeding safely.",
            "Remain stopped and wait for the pedestrian to safely reach the sidewalk before moving.",
            "Reduce speed significantly and be prepared to stop. Alert them with a gentle horn tap if needed.",
        ],
        "rejected": [
            "Proceed at normal speed since the light is green — the pedestrian should wait.",
            "Swerve around the child and continue at speed to avoid disrupting traffic flow.",
            "Maintain speed; pedestrians should look before stepping onto the road.",
            "Proceed quickly before the pedestrian steps off the curb.",
            "Honk to hurry them up and proceed once there is a gap.",
            "Accelerate through before they block your path completely.",
            "Maintain speed and expect the pedestrian to move out of the way.",
        ],
    },
    # 2. Speed management
    {
        "category": "speed",
        "prompts": [
            "You are driving at 60 mph in a school zone that now shows 20 mph. What should you do?",
            "Traffic ahead is slowing rapidly on the highway. What is the safest response?",
            "You are driving in heavy rain on the highway at the posted speed of 70 mph. What should you do?",
            "You enter a construction zone with a speed limit reduction. What do you do?",
            "You are driving downhill on a steep grade. What speed adjustment is appropriate?",
            "Traffic is stop-and-go on a freeway. What is the best approach?",
            "You are in a residential neighborhood with children playing nearby. What speed is appropriate?",
        ],
        "chosen": [
            "Immediately and progressively reduce speed to 20 mph, complying with the school zone limit to protect children.",
            "Gradually apply the brakes to reduce speed smoothly, increase following distance, and be ready to stop.",
            "Reduce speed well below the posted limit — roads are much more dangerous when wet. Increase following distance significantly.",
            "Reduce speed to match the posted construction zone limit, stay alert for workers, and maintain increased following distance.",
            "Gently apply brakes to regulate speed, use engine braking, and maintain safe following distance behind vehicles ahead.",
            "Maintain safe following distance, avoid frequent lane changes, and match the flow of traffic patiently.",
            "Slow to 15 mph or less, scan for children near the road, and be ready to stop at any moment.",
        ],
        "rejected": [
            "Maintain highway speed and slow down only if you see police.",
            "Tailgate the car in front so you have a shorter gap to manage.",
            "Keep the same speed — modern tires handle rain well enough at highway speeds.",
            "Maintain normal speed since construction workers are behind barriers.",
            "Accelerate to maintain momentum going downhill.",
            "Aggressively change lanes to find the fastest-moving lane.",
            "Drive at the full residential speed limit since children should stay out of the road.",
        ],
    },
    # 3. Lane changes
    {
        "category": "lane_change",
        "prompts": [
            "You want to change lanes on the highway. How should you proceed?",
            "A car is merging onto the freeway from an on-ramp in your lane. What do you do?",
            "You need to make a last-minute lane change to take your exit. What should you do?",
            "A large truck is drifting slightly into your lane. What is the safest response?",
            "You are in the middle lane and a vehicle on your right is signaling to move into your lane. What do you do?",
            "You want to pass a slow-moving vehicle. When is it safe to do so?",
            "You are driving in fog and need to change lanes. How should you proceed?",
        ],
        "chosen": [
            "Check mirrors, signal for at least 3 seconds, check blind spot, then smoothly move into the lane when clear.",
            "Ease off the accelerator or move to the left lane if safe to do so, giving the merging vehicle space to enter.",
            "Miss the exit and use the next available U-turn or exit — sudden late lane changes cause accidents.",
            "Gently steer away from the truck, increase following distance, and move to a safer lane when possible.",
            "Allow the vehicle to merge safely by maintaining or slightly reducing speed to create a gap.",
            "Only pass when you can see far enough ahead, use the left lane, signal, pass quickly, and return to the right.",
            "Reduce speed further, signal early, check mirrors and blind spot carefully before changing lanes slowly.",
        ],
        "rejected": [
            "Quickly cut across without signaling if you see a gap.",
            "Hold your position and force the merging car to slow down — you have the right of way.",
            "Cut quickly across two lanes to make your exit — it is worth the risk.",
            "Hold your lane position and hope the truck corrects itself.",
            "Speed up to prevent the other vehicle from merging.",
            "Pass on the right to avoid signaling and moving to the left lane.",
            "Change lanes at normal speed since the fog is not that thick.",
        ],
    },
    # 4. Weather conditions
    {
        "category": "weather",
        "prompts": [
            "Visibility drops sharply due to dense fog. How do you adjust your driving?",
            "You encounter black ice on the road. What is the correct response?",
            "A sudden thunderstorm makes the road very slippery. What should you do?",
            "You are driving in a blizzard with near-zero visibility. What is the safest action?",
            "Strong crosswinds are buffeting your vehicle on a bridge. How do you respond?",
            "Sun glare is blinding you at a busy intersection. What do you do?",
            "Flooding is covering part of the road ahead. What is the appropriate action?",
        ],
        "chosen": [
            "Slow down significantly, use low-beam headlights and fog lights, increase following distance to at least 4 seconds.",
            "Do not brake suddenly. Steer gently to stay in your lane, ease off the accelerator, and let the car decelerate gradually.",
            "Reduce speed substantially, turn on headlights, increase following distance, and avoid sudden steering or braking inputs.",
            "Pull over at a safe location if possible, use hazard lights, wait for conditions to improve before continuing.",
            "Reduce speed, grip the steering wheel firmly with both hands, and steer smoothly to counteract wind gusts.",
            "Slow down, use the sun visor, leave extra space ahead, and if visibility is zero, stop safely off the road.",
            "Turn around and find an alternate route. Never drive into floodwater — even shallow water can sweep away a vehicle.",
        ],
        "rejected": [
            "Keep normal speed and use high-beam headlights to see further through the fog.",
            "Apply brakes firmly to stop as quickly as possible.",
            "Continue at normal speed — modern ABS handles slippery roads.",
            "Continue slowly since other drivers will see your headlights.",
            "Ignore the wind and maintain highway speed to get off the bridge faster.",
            "Accelerate through the intersection quickly to get past the glare.",
            "Drive carefully through the water at low speed — it does not look very deep.",
        ],
    },
    # 5. Following distance / tailgating
    {
        "category": "following_distance",
        "prompts": [
            "The car ahead brakes suddenly. You are very close behind. What should you do?",
            "You are on a highway at 70 mph. What following distance should you maintain?",
            "A vehicle cuts in front of you, leaving a very small gap. What do you do?",
            "You are following a large truck and cannot see the road ahead. What should you do?",
            "The vehicle ahead is driving erratically and unpredictably. What is the safest response?",
            "You are approaching a red light with a car stopped ahead. How close should you stop?",
            "You are driving in wet conditions behind another vehicle. How should you adjust following distance?",
        ],
        "chosen": [
            "Apply brakes firmly but smoothly, steer to an escape route if a collision is unavoidable, and brace safely.",
            "Maintain at least a 3-4 second gap, which provides roughly 300 feet at that speed to react and stop safely.",
            "Ease off the gas, increase following distance again, and stay calm — do not retaliate or tailgate.",
            "Increase the following distance to at least 4 seconds to allow visibility of the road ahead and extra stopping time.",
            "Increase your following distance significantly, stay back, and prepare to change lanes or stop if necessary.",
            "Stop far enough back to see the rear tires of the vehicle ahead on the pavement — about 6-10 feet minimum.",
            "Double your following distance to at least 6 seconds to account for longer stopping distances on wet pavement.",
        ],
        "rejected": [
            "Slam the brakes and lean on the horn to warn other drivers.",
            "Stay 1-2 car lengths back to avoid other vehicles cutting in front of you.",
            "Speed up and tailgate to pressure them to move over.",
            "Stay close behind the truck so other vehicles do not cut in between you.",
            "Flash your lights and honk to signal the driver to pull over.",
            "Stop right at their bumper to leave room for vehicles behind you.",
            "Maintain the same following distance since the posted speed limit has not changed.",
        ],
    },
    # 6. Traffic signals and signs
    {
        "category": "traffic_signals",
        "prompts": [
            "The traffic light turns yellow as you approach at 40 mph. What should you do?",
            "You arrive at a four-way stop at the same time as a vehicle on your right. Who goes first?",
            "A traffic signal is broken and showing no lights at an intersection. How do you proceed?",
            "You see a red light ahead but a police officer is waving you through. What do you do?",
            "You are about to run a red light by mistake. What should you do?",
            "A stop sign is partially obscured by foliage. What should you do?",
            "The light turns green but a car is still in the intersection clearing. What do you do?",
        ],
        "chosen": [
            "If you can stop safely before the line, do so. If you are already close and stopping would be unsafe, proceed cautiously.",
            "The vehicle on the right has the right of way at a four-way stop when arriving simultaneously.",
            "Treat the intersection as a four-way stop: come to a complete stop, yield, and proceed only when it is safe.",
            "Follow the officer's directions — they supersede traffic signals when directing traffic.",
            "Brake as firmly and safely as possible, even if it means stopping past the line rather than running the red light.",
            "Come to a full stop and yield, even if the sign is partially hidden. Assume it is a stop sign.",
            "Wait for the intersection to clear before proceeding, even though the light is green.",
        ],
        "rejected": [
            "Accelerate through since slowing down might cause a rear-end collision from the car behind.",
            "You go first since you arrived at nearly the same time and you are in a hurry.",
            "Proceed through at normal speed since there is no signal telling you to stop.",
            "Stop anyway — you never run red lights regardless of who is directing.",
            "Accelerate through and hope there is no cross traffic.",
            "Proceed at normal speed since you do not see a stop sign clearly.",
            "Accelerate through immediately since you have a green light.",
        ],
    },
    # 7. Emergency vehicles
    {
        "category": "emergency",
        "prompts": [
            "An ambulance with lights and sirens approaches from behind. What should you do?",
            "A fire truck is blocking an intersection during an emergency. What is the correct action?",
            "A police car pulls up behind you with lights flashing on the highway. What do you do?",
            "An emergency vehicle is approaching a red light from a cross street with sirens on. What do you do?",
            "You see an accident ahead with emergency vehicles on the scene. What should you do?",
            "A pedestrian is injured by the roadside and needs help. What do you do?",
            "You hear a siren but cannot identify which direction it is coming from. What should you do?",
        ],
        "chosen": [
            "Signal, move to the right side of the road, and stop until the ambulance has fully passed.",
            "Stop completely, do not block the intersection further, and follow any instructions from emergency personnel.",
            "Signal right, carefully pull to the right shoulder, and stop. Wait until the officer approaches or signals to continue.",
            "Stop before entering the intersection and yield to the emergency vehicle even if you have a green light.",
            "Slow down, move over one lane away from emergency vehicles if safe, and stay alert for workers and hazards.",
            "Pull safely off the road, call emergency services (911), and render assistance only if trained and safe to do so.",
            "Slow down, pull to the right, and stop until you can determine the source and it has passed.",
        ],
        "rejected": [
            "Speed up to get out of the way faster.",
            "Drive around the fire truck quickly to clear the area.",
            "Continue driving at the speed limit and move over slightly.",
            "Proceed through the intersection since you have the green light.",
            "Speed past quickly to avoid the scene.",
            "Stop in the middle of the road immediately to check on the person.",
            "Continue driving — slowing down will only create traffic problems.",
        ],
    },
    # 8. Night driving
    {
        "category": "night_driving",
        "prompts": [
            "You are driving at night on an unlit rural road. What precautions should you take?",
            "An oncoming car has its high beams on, blinding you. What should you do?",
            "You spot an animal on the road ahead at night. What is the correct response?",
            "Your headlights illuminate a slow-moving cyclist ahead. What do you do?",
            "You are drowsy while driving at night on a long highway trip. What should you do?",
            "Your headlights are dim and making it hard to see at night. What should you do?",
            "You are driving through a tunnel at night and exit into bright daylight. What should you do?",
        ],
        "chosen": [
            "Use high beams when no oncoming traffic is present, reduce speed to match your sight distance, and stay alert.",
            "Look to the right edge of the road to avoid being fully blinded, slow down, and do not flash your high beams.",
            "Slow down immediately, flash your headlights, be prepared to stop, and do not swerve dangerously.",
            "Move to the far left of your lane, reduce speed, pass carefully with extra clearance, and do not honk unnecessarily.",
            "Pull over at a safe, well-lit rest area and rest or sleep until you are alert enough to continue safely.",
            "Reduce speed to match visibility, pull over safely, and check/replace bulbs or visit a mechanic promptly.",
            "Slow down and keep eyes on the road ahead to adjust quickly to changed visibility conditions.",
        ],
        "rejected": [
            "Drive at normal highway speed since your headlights light up the road adequately.",
            "Flash your high beams at the oncoming driver repeatedly to tell them to dim their lights.",
            "Swerve sharply around the animal without reducing speed.",
            "Honk loudly and maintain your speed to scare the cyclist off the road.",
            "Turn up the radio to stay awake and drink energy drinks to push through.",
            "Drive at normal speed — dimmer lights are annoying but manageable.",
            "Accelerate quickly to clear the tunnel exit before other vehicles reach you.",
        ],
    },
]

# ── Generate dataset ─────────────────────────────────────────────────────────

def build_prompt(scenario_text: str) -> str:
    return f"<s>[INST] {SYSTEM}\n\nScenario: {scenario_text} [/INST]"


def generate_pairs():
    pairs = []
    for scenario_group in SCENARIOS:
        category = scenario_group["category"]
        prompts = scenario_group["prompts"]
        chosens = scenario_group["chosen"]
        rejecteds = scenario_group["rejected"]

        for prompt, chosen, rejected in zip(prompts, chosens, rejecteds):
            pairs.append({
                "category": category,
                "prompt": build_prompt(prompt),
                "chosen": chosen,
                "rejected": rejected,
            })

    # Shuffle for training
    random.shuffle(pairs)
    return pairs


def main():
    os.makedirs("dataset", exist_ok=True)
    output_path = "dataset/driving_preferences.jsonl"

    pairs = generate_pairs()
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Generated {len(pairs)} preference pairs -> {output_path}")

    # Category breakdown
    from collections import Counter
    cats = Counter(p["category"] for p in pairs)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
