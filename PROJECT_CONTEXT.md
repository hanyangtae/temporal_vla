# Project: Time & Uncertainty Aware OpenVLA

## 1. Research Goal
**Objective**: Fine-tune the OpenVLA model to predict not only robotic actions but also the **Task Duration** and **Variance (Uncertainty)**.
- **Base Model**: OpenVLA (7B) or SmolVLA.
- **Dataset**: BEHAVIOR-1K (OmniGibson based).
- **Method**: Add `[Duration]` and `[Variance]` tokens to the output vocabulary.

## 2. Hardware Environment & Constraints
This project operates across three different machines. The AI assistant must check the current environment and provide code optimized for it.

### **PC 1 (Home)** - Data Factory & Final Eval
- **GPU**: RTX 4090 (24GB)
- **Role**: 
  - Massive data generation (OmniGibson High-Quality Rendering).
  - Final quantitative evaluation & statistics.
- **Constraints**: 24GB is enough for 8-bit inference + Sim, but heavy training is slow.

### **PC 2 (Lab)** - Development & Quick Debug
- **GPU**: RTX 3060 (12GB)
- **Role**: 
  - **Main Coding Workspace**: Immediate visual feedback (NoVNC/Direct Monitor).
  - **Code Logic Verification**: Sanity checks for simulation & model architecture.
- **Constraints**: **CRITICAL VRAM LIMIT (12GB)**. 
  - Must use **4-bit Quantization** for OpenVLA.
  - Must use **Low-Quality Assets** for OmniGibson.

### **Server (Training Center)**
- **GPU**: A100 (80GB)
- **Role**: 
  - Full Fine-tuning / LoRA Training.
- **Constraints**: No display output (Headless only). Focus on compute.

---

## 3. Workflow Pipeline
1.  **Step 1 (PC 2 - Lab)**: Develop code for data loading & model architecture. Verify logic with small data (1~2 demos).
    *   *Action*: Write code, debug visuals, commit to Git.
2.  **Step 2 (PC 1 - Home)**: Pull code. Generate large-scale dataset (RLDS format) with Duration/Variance labels.
    *   *Action*: Run data generation scripts, SCP data to Server.
3.  **Step 3 (Server)**: Train the model using the generated dataset.
    *   *Action*: Run training script, save checkpoints, SCP best checkpoint to PC 2 & PC 1.
4.  **Step 4 (PC 2 - Lab)**: Quick visual validation of the trained checkpoint.
    *   *Action*: Load model (4-bit), run sim, visually verify robot behavior.
5.  **Step 5 (PC 1 - Home)**: Final large-scale evaluation.
    *   *Action*: Run evaluation episodes, calculate success rate & time error stats.

---

## 4. Instructions for AI Assistant
When starting a session, please:
1.  **Identify the current machine** (Ask user or check specs).
2.  **Adopt the specific persona/constraints** for that machine:
    - If **PC 2**: Optimize for VRAM (12GB). Always suggest 4-bit quantization and memory-saving tips.
    - If **PC 1**: Focus on rendering quality and batch processing performance.
    - If **Server**: Focus on multi-GPU/high-VRAM training scripts and headless setup.
3.  **Reference the Workflow**: Understand which step we are currently on based on the user's request.

