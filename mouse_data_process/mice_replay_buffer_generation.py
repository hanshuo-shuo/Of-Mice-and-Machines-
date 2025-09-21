from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
import math
import datetime
import cellworld_game as cwgame
import cellworld_gym as cwgym
import typing
import cellworld as cw
import os
import shapely as sp

# Base path for mouse experiment data
start_path = "BotEvade_experiments/MICE"


def experiment_files(start_path: str) -> typing.List[str]:
    """Yield paths to experiment JSON files in the given directory."""
    entries = os.listdir(start_path)
    for entry in entries:
        full_path = os.path.join(start_path, entry)
        if os.path.isdir(full_path):
            experiment_file = f"{entry}_experiment.json"
            experiment_file_path = os.path.join(full_path, experiment_file)
            if os.path.exists(experiment_file_path):
                print(experiment_file_path)
                yield experiment_file_path


def get_step_information(episode: cw.Episode,
                        cells: cw.Cell_group,
                        visibility: cwgame.Visibility,
                        step_interval: float = 0.25,
                        goal_location: typing.Tuple[float, float] = (1.0, 0.5),
                        puff_threshold: float = 0.1,
                        goal_threshold: float = 0.1,
                        puff_cool_down_time: float = .5):
    """
    Process episode data at fixed time intervals and track predator visibility.
    Returns observation tuples containing prey and predator state information.
    """
    prev_prey_location: typing.Tuple[float, float] = (0.0, 0.5)
    last_puff_timestamp: float = -puff_cool_down_time

    def get_observation(prey_step: typing.Optional[cw.Step],
                       predator_step: typing.Optional[cw.Step]) -> cwgym.BotEvadeObservation:
        """Create observation from prey and predator steps."""
        nonlocal prev_prey_location, last_puff_timestamp
        observation = cwgym.BotEvadeObservation()

        # Process prey information
        prey_location = prey_step.location.get_values()
        prey_direction = cwgame.direction(src=prev_prey_location, dst=prey_location)
        goal_distance = cwgame.distance(src=prey_location, dst=goal_location)
        
        observation[cwgym.BotEvadeObservation.Field.prey_x] = prey_step.location.x
        observation[cwgym.BotEvadeObservation.Field.prey_y] = prey_step.location.y
        observation[cwgym.BotEvadeObservation.Field.prey_direction] = math.radians(prey_direction)
        observation[cwgym.BotEvadeObservation.Field.prey_goal_distance] = goal_distance
        observation[cwgym.BotEvadeObservation.Field.finished] = goal_distance < goal_threshold

        # Process predator information
        last_puff_time_diff = prey_step.time_stamp - last_puff_timestamp
        predator_x, predator_y, predator_direction, predator_distance, puffed = 0, 0, 0, 1, False

        if predator_step:
            predator_location = predator_step.location.get_values()
            is_visible = visibility.line_of_sight(src=sp.Point(prey_location), 
                                                dst=sp.Point(predator_location))

            if is_visible:
                predator_distance = cwgame.distance(src=predator_location, dst=prey_location)
                puffed = predator_distance <= puff_threshold and last_puff_time_diff > puff_cool_down_time
                
                if puffed:
                    last_puff_timestamp = predator_step.time_stamp
                    
                predator_x = predator_step.location.x
                predator_y = predator_step.location.y
                predator_direction = predator_step.rotation

        observation[cwgym.BotEvadeObservation.Field.predator_x] = predator_x
        observation[cwgym.BotEvadeObservation.Field.predator_y] = predator_y
        observation[cwgym.BotEvadeObservation.Field.predator_direction] = math.radians(predator_direction)
        observation[cwgym.BotEvadeObservation.Field.predator_prey_distance] = predator_distance
        observation[cwgym.BotEvadeObservation.Field.puffed] = puffed

        # Handle puff cooldown
        puff_cooled_down = max(0, puff_cool_down_time - last_puff_time_diff)
        observation[cwgym.BotEvadeObservation.Field.puff_cooled_down] = puff_cooled_down

        prev_prey_location = prey_location
        return observation

    # Split trajectories by agent
    trajectories = episode.trajectories.split_by_agent()
    if "prey" not in trajectories:
        return

    prey_trajectory = trajectories["prey"]
    prey_step_index = 0
    prey_step = prey_trajectory[prey_step_index]
    step_time = prey_step.time_stamp

    # Handle predator trajectory
    has_predator = "predator" in trajectories
    if has_predator:
        predator_trajectory = trajectories["predator"]
        predator_step = predator_trajectory.get_step_by_time_stamp(step_time)
    else:
        predator_trajectory = cw.Trajectories()
        predator_step = None

    # Process observations at fixed intervals
    pre_observation = get_observation(prey_step=prey_step, predator_step=predator_step)
    while prey_step_index < len(prey_trajectory):
        step_time += step_interval

        while prey_step_index < len(prey_trajectory) and prey_trajectory[prey_step_index].time_stamp < step_time:
            prey_step_index += 1

        if prey_step_index == len(prey_trajectory):
            break

        prey_step = prey_trajectory[prey_step_index]
        if has_predator:
            predator_step = predator_trajectory.get_step_by_time_stamp(step_time)

        prey_action = cells.find(prey_step.location)
        post_observation = get_observation(prey_step=prey_step, predator_step=predator_step)

        yield pre_observation, prey_action, post_observation
        pre_observation = post_observation

        if post_observation[cwgym.BotEvadeObservation.Field.finished.value]:
            break


def parse_experiment_name(experiment_name):
    """Parse experiment filename into components."""
    import re
    parts = experiment_name.split("_")
    prefix = parts[0]
    phase_iteration = parts[-1]
    match = re.match(r'(\D+)(\d+)', phase_iteration)
    if not match:
        return None
    phase = match.group(1)
    iteration = int(match.group(2))
    occlusions = "%s_%s" % (parts[-3], parts[-2])
    subject = parts[-4]
    experiment_date = datetime.datetime.strptime("%s_%s" % (parts[1], parts[2]), "%Y%m%d_%H%M")
    return prefix, experiment_date, subject, occlusions, phase, iteration


# Process each experiment file
for experiment_file_path in experiment_files(start_path):
    filename = os.path.basename(experiment_file_path)
    print(filename)
    
    experiment_parts = parse_experiment_name(filename.replace("_experiment.json", ""))
    if not experiment_parts:
        continue
        
    prefix, experiment_date, subject, world_occlusions, phase, iteration = experiment_parts
    replay_buffer_path = os.path.join("BotEvade_replay_buffers/MICE", prefix, subject, phase, str(iteration).zfill(3))
    
    if not os.path.isdir(replay_buffer_path):
        os.makedirs(replay_buffer_path, exist_ok=True)
        
    # Load and process experiment data
    experiment = cw.Experiment.load_from_file(experiment_file_path)
    loader = cwgame.CellWorldLoader(experiment.occlusions)
    visibility = cwgame.Visibility(arena=loader.arena, occlusions=loader.occlusions)
    
    replay_buffer = []
    for episode_number, episode in enumerate(experiment.episodes):
        for pre_observation, action, post_observation in get_step_information(
                episode=episode,
                cells=loader.world.cells.free_cells(),
                visibility=visibility):
            print(pre_observation, action, post_observation)
            replay_buffer.append((tuple[pre_observation], action, tuple[post_observation]))
            if post_observation[cwgym.BotEvadeObservation.Field.finished.value]:
                break
        # episode_file_path = os.path.join(replay_buffer_path, f"episode_%03d.pkl" % episode_number)
        # with open(episode_file_path, "wb") as f:
        #     pickle.dump(replay_buffer, f)
