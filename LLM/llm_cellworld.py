import gymnasium
import cellworld_gym as cwg
import matplotlib.pyplot as plt
from cellworld import *
import numpy as np
import json
from PIL import Image
from cellworld_game.video import save_video_output
from openai import OpenAI
import base64
import os
import pandas as pd
import copy


PROMPT = """
You are controlling a prey agent in a predator-prey environment. Your goal is to reach the green goal point (1.0, 0.5) while avoiding the predator and obstacles.

Environment Details:
- You (red dot) must reach the goal (green dot) while avoiding the predator (blue dot)
- Black areas are obstacles/walls that you cannot pass through
- The predator (blue dot) is constantly moving and trying to catch you, and there is a larger blue circle indicating the puffed area around the predator
- If you can't see the predator, it means it's hidden behind obstacles
- The environment has a grid to help you locate positions (x and y coordinates from 0 to 1)
- Each move must have an L2 norm less than 0.2

Your response must be a JSON object with exactly this format:
{
  "move": [
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>}
  ],
  "thoughts": "<single line explaining your strategy>"
}

Rules for moves:
1. Provide exactly 1 move
2. Each move should be a small step (L2 norm of distance between your next position and current position < 0.2)

Example response:
{
  "move": [
    {"x": 0.20, "y": 0.45},
  ],
  "thoughts": "Moving toward the goal while avoiding obstacles and keeping distance from potential predator locations."
}
"""

def get_message(image_path, PROMPT, last_action=None):
    MODEL = "gpt-4o"
    APIKEY = ### YOUR OPENAI API KEY HERE ###
    client = OpenAI(api_key=APIKEY)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(image_path)
    
    messages = [
        {"role": "system", "content": PROMPT}
    ]
    
    if last_action:
        messages.append({
            "role": "system",
            "content": f"Your last action was: {last_action}"
        })
    
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "low"
                }
            }
        ]
    })

    completion = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=messages
    )
    return completion.choices[0].message.content

def draw_cellworld_frame(obs):
    world = World.get_from_parameters_names("hexagonal", "canonical", "21_05")
    Display(world, show_axes=True, animated=True)

    # Plot the goal point
    plt.plot(obs[0], obs[1], 'o', markersize=15, color='r')     
    plt.gca().add_patch(plt.Circle((1, 0.5), 0.05, color='g', fill=True, alpha=0.5))

    # Add labels for the goal and the observer's position
    plt.text(1, 0.5, "goal", fontsize=15, verticalalignment='bottom', color='g')
    plt.text(obs[0], obs[1], 'you', fontsize=15, verticalalignment='bottom', color='r')

    # Check if a predator exists and plot its position if so
    if obs[4] == 0:
        predator_exists = False
    else:
        predator_exists = True
        plt.plot(obs[3], obs[4], 'o', markersize=15, color='b')
        plt.text(obs[3], obs[4], 'predator', fontsize=15, verticalalignment='bottom', color='b')
        plt.gca().add_patch(plt.Circle((obs[3], obs[4]), 0.1, color='b', fill=True, alpha=0.5))

    # # Add the custom title within the plot area
    title_text = "Cellworld with you and predator" if predator_exists else \
        "Cellworld with you, but you don't know where the predator is"
    # add the title and location should be in the graph
    plt.text(0, 1, title_text, fontsize=15, verticalalignment='bottom', color='black')
    plt.xticks(np.arange(0, 1.05, 0.05))
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid(linestyle='--', linewidth=0.5)
    fig = plt.gcf()
    return fig


if __name__=="__main__":
    reward_function_normal = cwg.Reward({"puffed": -1, "finished": 1})
    env = gymnasium.make("CellworldBotEvade-v0",
                         world_name="21_05",
                         use_lppos=False,
                         use_predator=True,
                         max_step=300,
                         time_step=0.25,
                         render=True,
                         real_time=True,
                         reward_function=reward_function_normal,
                         action_type=cwg.BotEvadeEnv.ActionType.CONTINUOUS)
    
    num_episodes = 1
    for episode in range(5, num_episodes+1):
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        next_obs_list = []
        thoughts_list = []
        
        obs, _ = env.reset()
        fig = draw_cellworld_frame(obs)
        fig.savefig(f"frame/episode_{episode}_frame_0.png", dpi=64)
        plt.close(fig)
        
        temp_frame = env.model.view.get_screen()
        temp_frame = Image.fromarray(temp_frame)
        temp_frame.save(f"god_view/episode_{episode}_frame_0.png")
        save_video_output(env.model, f"video/episode_{episode}")
        
        step = 1
        last_action = None
        done = False
        truncated = False
        
        # Create episode-specific thought file
        with open(f"thought_episode_{episode}.txt", "w") as f:
            f.write("")
            
        while not (done or truncated):
            fig = draw_cellworld_frame(obs)
            temp_frame_path = f"frame/episode_{episode}_frame_{step}.png"
            fig.savefig(temp_frame_path, dpi=64)
            plt.close(fig)
            
            try:
                response = json.loads(get_message(temp_frame_path, PROMPT, last_action))
            except Exception as e:
                print(f"Error occurred in episode {episode}: {e}")
                break
                
            print(f"Episode {episode}, Step {step}:", response)
            thoughts = response["thoughts"]
            action = response["move"][0]
            
            # Store current state
            obs_list.append(copy.deepcopy(obs))
            action_list.append([action["x"], action["y"]])
            thoughts_list.append(thoughts)
            
            # Take action
            next_obs, reward, done, truncated, info = env.step(action=np.array([action["x"], action["y"]]))
            
            # Store transition
            reward_list.append(reward)
            done_list.append(done or truncated)
            next_obs_list.append(copy.deepcopy(next_obs))
            
            obs = next_obs
            
            fig = draw_cellworld_frame(obs)
            fig.savefig(f"frame/episode_{episode}_frame_{step}.png", dpi=64)
            plt.close(fig)
            
            temp_frame = env.model.view.get_screen()
            temp_frame = Image.fromarray(temp_frame)
            temp_frame.save(f"god_view/episode_{episode}_frame_{step}.png")
            
            last_action = json.dumps(response["move"])
            
            with open(f"thought_episode_{episode}.txt", "a") as f:
                f.write(f"Step {step}: {thoughts}\n")
                
            step += 1
        
        # Save episode data
        data = {
            "obs": obs_list,
            "action": action_list,
            "reward": reward_list,
            "done": done_list,
            "next_obs": next_obs_list,
            "thoughts": thoughts_list
        }
        df = pd.DataFrame(data)
        df.to_csv(f"trajectory_data_episode_{episode}.csv", index=False)