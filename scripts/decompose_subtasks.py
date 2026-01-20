#!/usr/bin/env python3
"""
LIBERO 데이터셋을 Subtask 단위로 분해

1. LLM을 사용해 Task → Subtasks 리스트 생성
2. Trajectory의 action을 분석해 move primitive 추출
3. LLM을 사용해 Trajectory → Subtask 경계 인덱스 할당
4. Subtask 단위로 time_to_go 레이블 추가

Usage:
    python scripts/decompose_subtasks.py \
        --input data/datasets/libero_goal \
        --output data/datasets/libero_goal_subtasks
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# OpenAI API (or use Anthropic Claude)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ============================================================================
# Prompts from the paper
# ============================================================================

SUBTASK_PROPOSAL_PROMPT = """You are given a high-level robotic task description. Your job is to decompose this task into a minimal set of formal subtasks that a robot must perform to complete the task.

Rules:
1. Minimal Subtask Decomposition
- Focus on the necessary and sufficient actions to complete the task.
- Do not over-decompose — prefer atomic but essential steps.
- Ask: What does the robot must do to succeed, regardless of variations in execution?

2. Object-Centric Reasoning
- Use the presence and spatial arrangement of objects as key indicators.
- Mention relative spatial cues (e.g., above the drawer handle) if implied in the task.

3. Skillset Usage and Formal Language
- Each subtask must begin with one of these actions (verbs):
  1) "Move the gripper ..."
  2) "Rotate the gripper ..."
  3) "Open the gripper ..."
  4) "Close the gripper ..."
- Keep language precise, formal, and robotic.
- As a rule of thumb:
  1) Move/Rotate is often followed by Close to grasp.
  2) Grasped objects are then moved, followed by Open to release.

Input:
Task: {task}

Output Format (JSON only, no other text):
{{"subtasks": ["<subtask_1>", "<subtask_2>", ...], "reasoning": "<brief explanation>"}}
"""

SUBTASK_BOUNDARY_PROMPT = """You are an expert reinforcement learning researcher. You have trained an optimal policy to control a robotic arm, which successfully completed a task as specified in natural language. The robot executed a sequence of actions to complete this task. Each action is recorded as a step in a trajectory.

Input:
1. Task: {task}
2. Subtasks list (EXACTLY {num_subtasks} subtasks): {subtasks}
3. Trajectory features: Each entry in the dictionary corresponds to a single step on the trajectory and describes the move that is about to be executed.

trajectory_features = {trajectory_features}

Instruction:
Decompose the entire trajectory into EXACTLY {num_subtasks} subtasks (as defined above) and assign a start and end step index to each subtask.

**CRITICAL: You MUST use EXACTLY the {num_subtasks} subtasks provided above. Do NOT create more or fewer subtasks.**

The goal is to produce a mapping using the EXACT subtask names from the list:
Labeled_dict = {{"<exact_subtask_name_1>": [start_idx, end_idx], "<exact_subtask_name_2>": [start_idx, end_idx], ...}}

Rules:
1. STRICT Subtask Count
- You MUST output EXACTLY {num_subtasks} subtasks, no more, no less.
- Use the EXACT subtask names from the provided list as dictionary keys.
- Do NOT invent new subtask names like "subtask_5", "subtask_6", etc.

2. Noisy Labels
- The trajectory data contains noise. Apply the following rules carefully:
- "stop" labels often appear even when the robot is still moving.
- Treat "stop" as movement if it appears between meaningful motion steps.
- Do NOT segment a new subtask just because you see a "stop".
- "open/close gripper" labels that persist for very short durations may be noise. However, gripper actions are generally more reliable than motion primitives.

3. Robust Labeling Strategies
- Brief, short, inconsistent movement descriptions that conflict with prior and subsequent steps are likely to be noise.
- Cross-reference other movement labels to decide whether a short-duration label is meaningful or just noise.

4. Exhaustive Coverage
- You must not skip any steps. Every step in the trajectory should be assigned to exactly one subtask.
- There should be no gaps in the index ranges. Subtasks must be labeled sequentially and cover the entire range of the trajectory indices (0 to {max_step_idx}).

Output Format (JSON only, no other text):
{{"labeled_dict": {{"<exact_subtask_name>": [start_idx, end_idx], ...}}, "reasoning": "<brief explanation>"}}

Example for {num_subtasks} subtasks:
{{"labeled_dict": {{"{example_subtask}": [0, 50], ...}}, "reasoning": "..."}}
"""


# ============================================================================
# Action Primitive Extraction
# ============================================================================

def extract_move_primitive(action: np.ndarray, prev_action: np.ndarray = None,
                           gripper_state: float = None, prev_gripper_state: float = None,
                           move_threshold: float = 0.005,
                           rot_threshold: float = 0.02) -> str:
    """
    Extract movement primitive from action.
    
    action: [dx, dy, dz, drx, dry, drz, gripper_cmd] (7-dim)
    gripper_state: actual gripper state from observation (more reliable than action)
    
    Returns one of:
    - "move_forward", "move_backward", "move_left", "move_right", "move_up", "move_down"
    - "rotate_cw", "rotate_ccw"  
    - "open_gripper", "close_gripper", "gripper_opening", "gripper_closing"
    - "stop"
    
    CycleVLA uses gripper state changes as primary segmentation cues (Appendix B.2)
    """
    pos_delta = action[:3]  # dx, dy, dz
    rot_delta = action[3:6]  # drx, dry, drz
    gripper_cmd = action[6]  # gripper command
    
    # Gripper state detection (most reliable for subtask boundaries)
    # Use actual gripper state if available, otherwise use command
    if gripper_state is not None and prev_gripper_state is not None:
        gripper_change = gripper_state - prev_gripper_state
        if gripper_change > 0.1:
            return "gripper_opening"
        elif gripper_change < -0.1:
            return "gripper_closing"
    elif prev_action is not None:
        prev_gripper_cmd = prev_action[6]
        gripper_change = gripper_cmd - prev_gripper_cmd
        if gripper_change > 0.3:
            return "open_gripper"
        elif gripper_change < -0.3:
            return "close_gripper"
    
    # Check rotation
    rot_magnitude = np.linalg.norm(rot_delta)
    if rot_magnitude > rot_threshold:
        # Determine primary rotation axis
        abs_rot = np.abs(rot_delta)
        max_rot_idx = np.argmax(abs_rot)
        if max_rot_idx == 2:  # z-axis rotation
            return "rotate_ccw" if rot_delta[2] > 0 else "rotate_cw"
        elif max_rot_idx == 0:  # x-axis rotation (pitch)
            return "rotate_pitch_up" if rot_delta[0] > 0 else "rotate_pitch_down"
        else:  # y-axis rotation (roll)
            return "rotate_roll_cw" if rot_delta[1] > 0 else "rotate_roll_ccw"
    
    # Check translation
    pos_magnitude = np.linalg.norm(pos_delta)
    if pos_magnitude > move_threshold:
        # Find dominant direction
        abs_delta = np.abs(pos_delta)
        max_idx = np.argmax(abs_delta)
        
        if max_idx == 0:  # x-axis
            return "move_forward" if pos_delta[0] > 0 else "move_backward"
        elif max_idx == 1:  # y-axis
            return "move_right" if pos_delta[1] > 0 else "move_left"
        else:  # z-axis
            return "move_up" if pos_delta[2] > 0 else "move_down"
    
    return "stop"


# def detect_gripper_events(gripper_states: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, str]]:
#     """
#     Detect gripper open/close events from gripper state sequence.
    
#     Returns list of (step_idx, event_type) where event_type is "open" or "close"
    
#     CycleVLA Appendix B.2: Gripper state changes are the most reliable cue for subtask boundaries
#     """
#     events = []
    
#     # Detect transitions
#     for i in range(1, len(gripper_states)):
#         prev_open = gripper_states[i-1] > threshold
#         curr_open = gripper_states[i] > threshold
        
#         if not prev_open and curr_open:
#             events.append((i, "gripper_open"))
#         elif prev_open and not curr_open:
#             events.append((i, "gripper_close"))
    
#     return events


def extract_trajectory_features(actions: np.ndarray, 
                                 gripper_states: np.ndarray = None) -> Dict[int, str]:
    """
    Extract move primitives for entire trajectory.
    
    Args:
        actions: (N, 7) action sequence
        gripper_states: (N,) or (N, 2) gripper state observations (optional but recommended)
    
    Returns:
        Dict mapping step index to movement primitive string
    """
    features = {}
    prev_action = None
    prev_gripper = None
    
    # Handle gripper states shape
    if gripper_states is not None:
        if len(gripper_states.shape) > 1:
            gripper_states = gripper_states[:, 0]  # Use first gripper finger
    
    for i, action in enumerate(actions):
        gripper = gripper_states[i] if gripper_states is not None else None
        primitive = extract_move_primitive(
            action, prev_action, 
            gripper_state=gripper, 
            prev_gripper_state=prev_gripper
        )
        features[i] = primitive
        prev_action = action
        prev_gripper = gripper
    
    return features


def simplify_trajectory_features(features: Dict[int, str], window_size: int = 5) -> Dict[int, str]:
    """
    Simplify trajectory by merging consecutive same primitives
    Returns a sampled version for LLM (to reduce token count)
    """
    # Sample every N steps for LLM
    sampled = {}
    step = max(1, len(features) // 50)  # Max 50 entries for LLM
    
    for i in range(0, len(features), step):
        sampled[i] = features[i]
    
    # Always include last step
    sampled[len(features) - 1] = features[len(features) - 1]
    
    return sampled


# ============================================================================
# LLM API Calls
# ============================================================================

def call_openai(prompt: str, api_key: str, model: str = "gpt-5.1") -> str:
    """Call OpenAI API
    
    Available models (2026):
    - gpt-4o: Best quality, good for complex reasoning
    - gpt-4o-mini: Faster, cheaper
    - gpt-5.1: Latest, best for structured outputs (if available)
    """
    client = openai.OpenAI(api_key=api_key)
    
    # Configure parameters based on model type
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Handle model-specific parameters (o1, gpt-5.1 etc need max_completion_tokens)
    if model.startswith("o1") or model == "gpt-5.1":
        kwargs["max_completion_tokens"] = 2000
        # o1 models might have strict temperature requirements (often 1.0)
        if model.startswith("o1"):
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = 0.1
    else:
        # Legacy/Standard models (gpt-4o, gpt-3.5)
        kwargs["max_tokens"] = 2000
        kwargs["temperature"] = 0.1

    try:
        response = client.chat.completions.create(**kwargs)
    except openai.BadRequestError as e:
        # Fallback if parameter mismatch occurs
        if "max_completion_tokens" in str(e) and "max_tokens" in kwargs:
            kwargs.pop("max_tokens")
            kwargs["max_completion_tokens"] = 2000
            response = client.chat.completions.create(**kwargs)
        elif "max_tokens" in str(e) and "max_completion_tokens" in kwargs:
            kwargs.pop("max_completion_tokens")
            kwargs["max_tokens"] = 2000
            response = client.chat.completions.create(**kwargs)
        else:
            raise e
    
    return response.choices[0].message.content




def call_llm(prompt: str, api_key: str, provider: str = "openai", model: str = None) -> str:
    """Call LLM API"""
    if provider == "openai":
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        model = model or "gpt-4o"
        return call_openai(prompt, api_key, model=model)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def parse_json_response(response: str) -> dict:
    """Parse JSON from LLM response"""
    # Try to extract JSON from response
    response = response.strip()
    
    # Remove markdown code blocks if present
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join(lines[1:-1])
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON in response
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from response: {response}")


# ============================================================================
# Main Decomposition Pipeline
# ============================================================================

def get_subtasks_for_task(task: str, api_key: str, provider: str = "openai", 
                          model: str = None, cache: dict = None) -> List[str]:
    """
    Step 1: Get subtasks list for a task using LLM
    """
    if cache and task in cache:
        return cache[task]
    
    prompt = SUBTASK_PROPOSAL_PROMPT.format(task=task)
    response = call_llm(prompt, api_key, provider, model=model)
    result = parse_json_response(response)
    
    subtasks = result.get("subtasks", [])
    
    if cache is not None:
        cache[task] = subtasks
    
    return subtasks


def get_subtask_boundaries(task: str, subtasks: List[str], 
                           trajectory_features: Dict[int, str],
                           api_key: str, provider: str = "openai",
                           model: str = None) -> Dict[str, List[int]]:
    """
    Step 2: Get subtask boundaries for a trajectory using LLM
    Returns dict with actual subtask names as keys
    """
    # Simplify features for LLM
    simplified = simplify_trajectory_features(trajectory_features)
    max_step_idx = max(trajectory_features.keys())
    
    prompt = SUBTASK_BOUNDARY_PROMPT.format(
        task=task,
        subtasks=json.dumps(subtasks),
        num_subtasks=len(subtasks),
        trajectory_features=json.dumps(simplified, indent=2),
        max_step_idx=max_step_idx,
        example_subtask=subtasks[0] if subtasks else "subtask_1"
    )
    
    response = call_llm(prompt, api_key, provider, model=model)
    result = parse_json_response(response)
    
    labeled_dict = result.get("labeled_dict", {})
    
    # Map generic keys (subtask_1, subtask_2, ...) to actual subtask names
    # AND filter out any extra subtasks beyond the defined count
    mapped_dict = {}
    for key, bounds in labeled_dict.items():
        # Try to extract subtask index from key
        if key.startswith("subtask_"):
            try:
                idx = int(key.split("_")[1]) - 1  # subtask_1 -> index 0
                if 0 <= idx < len(subtasks):
                    mapped_dict[subtasks[idx]] = bounds
                # Skip subtasks beyond defined count (subtask_5, subtask_6, etc.)
            except (ValueError, IndexError):
                pass
        elif key in subtasks:
            # Key is already the actual subtask name
            mapped_dict[key] = bounds
        # Skip any unknown keys
    
    # Ensure we have all defined subtasks (fill missing ones)
    if len(mapped_dict) < len(subtasks):
        # Some subtasks are missing, need to handle this
        pass
    
    return mapped_dict


def interpolate_boundaries(labeled_dict: Dict[str, List[int]], 
                           total_steps: int) -> Dict[str, List[int]]:
    """
    Interpolate boundaries to cover all steps (LLM might have gaps)
    """
    if not labeled_dict:
        return {"full_task": [0, total_steps - 1]}
    
    # Sort subtasks by start index
    sorted_subtasks = sorted(labeled_dict.items(), key=lambda x: x[1][0])
    
    result = {}
    prev_end = -1
    
    for i, (subtask, (start, end)) in enumerate(sorted_subtasks):
        # Fill gap if any
        if start > prev_end + 1:
            start = prev_end + 1
        
        # Ensure end doesn't exceed total
        end = min(end, total_steps - 1)
        
        # If this is the last subtask, extend to end
        if i == len(sorted_subtasks) - 1:
            end = total_steps - 1
        
        result[subtask] = [start, end]
        prev_end = end
    
    return result


# ============================================================================
# Data Processing
# ============================================================================

def process_demo(demo_data: h5py.Group, task: str, subtasks: List[str],
                 labeled_dict: Dict[str, List[int]], fps: float = 10.0) -> List[dict]:
    """
    Process a single demo and create subtask-level data entries
    
    Each entry contains:
    - subtask_name: The actual subtask description (e.g., "Move the gripper to the drawer handle")
    - subtask_idx: Index of the subtask in the sequence (0, 1, 2, ...)
    - time_to_go_subtask: Time remaining to complete current subtask
    """
    actions = demo_data['actions'][:]
    n_steps = actions.shape[0]
    
    entries = []
    
    # Sort subtasks by start index to get proper ordering
    sorted_subtasks = sorted(labeled_dict.items(), key=lambda x: x[1][0])
    
    for subtask_idx, (subtask_name, (start_idx, end_idx)) in enumerate(sorted_subtasks):
        subtask_length = end_idx - start_idx + 1
        
        for step in range(start_idx, end_idx + 1):
            # Time to complete THIS subtask
            steps_remaining = end_idx - step
            time_to_go_subtask = steps_remaining / fps
            
            entry = {
                "global_step": step,
                "subtask_name": subtask_name,  # Actual subtask description for VLA
                "subtask_idx": subtask_idx,    # Index in sequence
                "subtask_start": start_idx,
                "subtask_end": end_idx,
                "subtask_length": subtask_length,
                "step_in_subtask": step - start_idx,
                "time_to_go_subtask": time_to_go_subtask,  # Time to complete subtask
                "time_to_go_full": (n_steps - 1 - step) / fps,  # Time to complete full task
                "is_subtask_end": step == end_idx,  # Flag for subtask completion
                "total_subtasks": len(sorted_subtasks),
            }
            entries.append(entry)
    
    return entries


def process_hdf5_file(hdf5_path: Path, output_dir: Path, task_name: str,
                      api_key: str, provider: str = "openai",
                      model: str = None, subtask_cache: dict = None, 
                      fps: float = 10.0, max_demos: int = None) -> dict:
    """
    Process a single HDF5 file and create subtask-decomposed dataset
    """
    print(f"\n📂 Processing: {hdf5_path.name}")
    
    # Get subtasks for this task (cached)
    subtasks = get_subtasks_for_task(task_name, api_key, provider, model=model, cache=subtask_cache)
    print(f"   Subtasks ({len(subtasks)}개): {subtasks}")
    
    with h5py.File(hdf5_path, 'r') as f:
        demos = sorted(
            [k for k in f['data'].keys() if k.startswith('demo')],
            key=lambda x: int(x.split('_')[1])
        )
        
        # Limit demos if max_demos is set
        if max_demos is not None:
            demos = demos[:max_demos]
            print(f"   (Testing mode: processing {len(demos)} demos only)")
        
        all_entries = []
        demo_metadata = []
        
        for demo_idx, demo_name in enumerate(tqdm(demos, desc="   Demos")):
            demo = f['data'][demo_name]
            actions = demo['actions'][:]
            n_steps = actions.shape[0]
            # Get gripper states from observations (more reliable than action commands)
            gripper_states = None
            if 'obs' in demo and 'gripper_states' in demo['obs']:
                gripper_states = demo['obs']['gripper_states'][:]
            
            # Extract trajectory features with gripper state info
            traj_features = extract_trajectory_features(actions, gripper_states)
            
            # Detect gripper events for better boundary detection
            # gripper_events = []
            # if gripper_states is not None:
            #     gripper_events = detect_gripper_events(
            #         gripper_states[:, 0] if len(gripper_states.shape) > 1 else gripper_states
            #     )
            
            # Get subtask boundaries from LLM
            try:
                labeled_dict = get_subtask_boundaries(
                    task_name, subtasks, traj_features, api_key, provider, model=model
                )
                # Interpolate to ensure full coverage
                labeled_dict = interpolate_boundaries(labeled_dict, n_steps)
            except Exception as e:
                print(f"   ⚠️ LLM failed for {demo_name}: {e}")
                # Fallback: treat entire trajectory as one subtask
                labeled_dict = {"full_task": [0, n_steps - 1]}
            
            # Process demo with subtask info
            entries = process_demo(demo, task_name, subtasks, labeled_dict, fps)
            
            # Add demo index
            for entry in entries:
                entry["demo_idx"] = demo_idx
            
            all_entries.extend(entries)
            
            demo_metadata.append({
                "demo_idx": demo_idx,
                "n_steps": n_steps,
                "subtasks": labeled_dict,
            })
    
    # Save metadata
    metadata = {
        "task": task_name,
        "subtasks": subtasks,
        "n_demos": len(demos),
        "n_entries": len(all_entries),
        "demos": demo_metadata,
    }
    
    task_output_dir = output_dir / task_name.replace(" ", "_")
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(task_output_dir / "subtask_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    with open(task_output_dir / "subtask_entries.jsonl", "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"   ✅ Saved {len(all_entries)} entries to {task_output_dir}")
    
    return metadata


# ============================================================================
# Main
# ============================================================================

def analyze_results(output_dir: Path) -> List[dict]:
    """
    Analyze decomposition results and identify abnormal cases
    """
    analysis = []
    
    for task_dir in output_dir.iterdir():
        if not task_dir.is_dir():
            continue
        
        metadata_file = task_dir / "subtask_metadata.json"
        if not metadata_file.exists():
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        task_name = metadata["task"]
        expected_subtasks = metadata["subtasks"]
        n_expected = len(expected_subtasks)
        
        for demo in metadata["demos"]:
            demo_idx = demo["demo_idx"]
            n_steps = demo["n_steps"]
            subtasks = demo["subtasks"]
            
            # Check for issues
            issues = []
            
            # Issue 1: Wrong number of subtasks
            if len(subtasks) != n_expected:
                issues.append(f"Wrong count: {len(subtasks)} vs {n_expected}")
            
            # Issue 2: Numbered subtasks instead of names
            for key in subtasks.keys():
                if key.startswith("subtask_") or key == "full_task":
                    issues.append(f"Numbered subtask: {key}")
                    break
            
            # Issue 3: Overlapping or very short segments
            sorted_subs = sorted(subtasks.items(), key=lambda x: x[1][0])
            for i, (name, (start, end)) in enumerate(sorted_subs):
                length = end - start + 1
                if length <= 1:
                    issues.append(f"Too short: {name} ({length} steps)")
                
                # Check overlap with next
                if i < len(sorted_subs) - 1:
                    next_start = sorted_subs[i+1][1][0]
                    if end >= next_start:
                        issues.append(f"Overlap at {name}")
            
            # Issue 4: Gaps in coverage
            prev_end = -1
            for name, (start, end) in sorted_subs:
                if start > prev_end + 1:
                    issues.append(f"Gap before {name}")
                prev_end = end
            
            status = "✅ OK" if not issues else "❌ " + "; ".join(issues)
            
            analysis.append({
                "task": task_name,
                "demo_idx": demo_idx,
                "n_steps": n_steps,
                "n_subtasks": len(subtasks),
                "expected_subtasks": n_expected,
                "status": status,
                "issues": "; ".join(issues) if issues else "",
                "subtask_names": list(subtasks.keys())
            })
    
    return analysis


def save_analysis_to_excel(analysis: List[dict], output_path: Path):
    """
    Save analysis results to Excel file
    """
    try:
        import pandas as pd
        
        df = pd.DataFrame(analysis)
        
        # Reorder columns
        columns = ["task", "demo_idx", "n_steps", "n_subtasks", "expected_subtasks", 
                   "status", "issues", "subtask_names"]
        df = df[columns]
        
        # Save to Excel
        df.to_excel(output_path, index=False, sheet_name="Analysis")
        print(f"📊 Analysis saved to {output_path}")
        
        # Print summary
        total = len(df)
        ok_count = len(df[df["status"] == "✅ OK"])
        error_count = total - ok_count
        
        print(f"\n📈 Summary:")
        print(f"   Total demos: {total}")
        print(f"   ✅ OK: {ok_count} ({100*ok_count/total:.1f}%)")
        print(f"   ❌ Issues: {error_count} ({100*error_count/total:.1f}%)")
        
    except ImportError:
        print("⚠️ pandas not installed. Saving as CSV instead.")
        import csv
        with open(output_path.with_suffix('.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=analysis[0].keys())
            writer.writeheader()
            writer.writerows(analysis)
        print(f"📊 Analysis saved to {output_path.with_suffix('.csv')}")


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Decompose LIBERO data into subtasks")
    parser.add_argument("--input", type=str, 
                        default="data/datasets/libero_goal",
                        help="Input directory containing HDF5 files")
    parser.add_argument("--output", type=str, 
                        default="data/datasets/libero_goal_subtasks",
                        help="Output directory for subtask data")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic"],
                        help="LLM provider")
    parser.add_argument("--model", type=str, default="gpt-5.1",
                        help="LLM model (gpt-4o, gpt-4o-mini, gpt-5.1, claude-3-5-sonnet)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="FPS for time calculation")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to process (default: all)")
    parser.add_argument("--max_demos", type=int, default=None,
                        help="Maximum number of demos to process per task (for testing)")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze existing results without processing")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze only mode
    if args.analyze_only:
        print("📊 Analyzing existing results...")
        analysis = analyze_results(output_dir)
        save_analysis_to_excel(analysis, output_dir / "analysis_results.xlsx")
        return
    
    # Get API key from environment variables
    if args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if api_key is None:
        raise ValueError(f"API key not found in environment variables. Please set {args.provider.upper()}_API_KEY in your .env file.")
    
    input_dir = Path(args.input)
    
    # Find HDF5 files
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    
    if args.tasks:
        # Filter to specific tasks
        hdf5_files = [f for f in hdf5_files 
                      if any(t in f.stem for t in args.tasks)]
    
    print(f"🔍 Found {len(hdf5_files)} HDF5 files")
    print(f"🤖 Using model: {args.model}")
    
    # Cache for subtasks (same task = same subtasks)
    subtask_cache = {}
    
    all_metadata = []
    
    for hdf5_path in hdf5_files:
        # Extract task name from filename
        task_name = hdf5_path.stem.replace("_demo", "").replace("_", " ")
        
        try:
            metadata = process_hdf5_file(
                hdf5_path, output_dir, task_name,
                api_key, args.provider, model=args.model,
                subtask_cache=subtask_cache, fps=args.fps,
                max_demos=args.max_demos
            )
            all_metadata.append(metadata)
        except Exception as e:
            print(f"❌ Failed to process {hdf5_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save global metadata
    with open(output_dir / "global_metadata.json", "w") as f:
        json.dump({
            "tasks": [m["task"] for m in all_metadata],
            "subtask_definitions": subtask_cache,
            "total_demos": sum(m["n_demos"] for m in all_metadata),
            "total_entries": sum(m["n_entries"] for m in all_metadata),
            "model": args.model,
        }, f, indent=2)
    
    print(f"\n🎉 Processing complete! Output saved to {output_dir}")
    
    # Auto-analyze results
    print("\n📊 Analyzing results...")
    analysis = analyze_results(output_dir)
    save_analysis_to_excel(analysis, output_dir / "analysis_results.xlsx")


if __name__ == "__main__":
    main()
