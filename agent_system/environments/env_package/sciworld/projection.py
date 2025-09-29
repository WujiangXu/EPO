from typing import List
import re

def sciworld_projection(actions: List[str], available_actions=None, meta_think=False):
    valids = [0] * len(actions)
    action_available = [False] * len(actions)
    processed_actions = []

    for i in range(len(actions)):
        original_str = actions[i]
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = original_str.find(start_tag)
        end_idx = original_str.find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                processed_actions.append(original_str[-20:])
                continue
            extracted_action = original_str[start_idx + len(start_tag):end_idx].strip()
            processed_actions.append(extracted_action)
            valids[i] = 1
            env_available_actions = available_actions[i]
            if extracted_action in env_available_actions:
                action_available[i] = True
        except:
            processed_actions.append(original_str[-20:])
        if meta_think:
            if ("<planning>" not in original_str or "</planning>" not in original_str) and \
               ("<explore>" not in original_str or "</explore>" not in original_str) and \
               ("<reflection>" not in original_str or "</reflection>" not in original_str) and \
               ("<monitor>" not in original_str or "</monitor>" not in original_str):
                valids[i] = 0
        else:
            think_start_idx = original_str.find("<think>")
            think_end_idx = original_str.find("</think>")
            if think_start_idx == -1 or think_end_idx == -1:
                valids[i] = 0
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return processed_actions, valids, action_available
