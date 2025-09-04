# ========================================
# Multi-Model Answer Accuracy Analysis and Radar Chart Visualization (English Version - 6 Dimensions)
# ========================================

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model configuration
MODELS_base = {
    "GPT-4o": "gpt_answer_4o",
    "Gemini-2.5-Pro": "gemini-2.5-pro-preview-06-05",
    "Cladue-3-5-sonnect": "claude_simplified_answer",
    "Qwen-VL-MAX": "qwen_vl_max_answer",
    "GPT-4o-2": "gpt-4o",
}

MODELS_ours = {
    "qwen-7b": "origin",
    "origin3-s-sft": "origin3-s-sft",
    "loragen-sft_0.5_2": "loragen-sft_0.5_2",
    "cn_fixed":"cn_fixed",
    "cn-loragen-mixed-sft\checkpoint-14000":"checkpoint-14000",
    "cn_lora_fixed":"cn_lroa_fixed",
}
results_path = [
        '/data/xdd/LLaMA-Factory/saves/cultrual_heritage/qwen2_5vl-7b/origin/result_new_test.json',
        '/data/xdd/LLaMA-Factory/saves/cultrual_heritage/qwen2_5vl-7b/lora/origin_v3-s-sft/result_new_test.json',
        '/data/xdd/LLaMA-Factory/saves/cultrual_heritage/qwen2_5vl-7b/lora/loragen-sft_0.5_2/result_new_test.json',
        '/data/xdd/LLaMA-Factory/saves/cultrual_heritage/qwen2_5vl-7b/lora/loragen_fixed/result_new_test.json',
        '/data/xdd/LLaMA-Factory/saves/cultrual_heritage/qwen2_5vl-7b/lora/cn-loragen-mixed-sft/result_new_test.json',
        '/data/xdd/LLaMA-Factory/saves/cultrual_heritage/qwen2_5vl-7b/lora/cn_lora_fixed/result_new_test.json',
           ]

#MODELS = {**MODELS_base, **MODELS_ours}
MODELS = {**MODELS_ours}

# Reset font settings for English
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 1. Load data and calculate accuracy
print("🔍 Analyzing Multi-Model Answer Accuracy...")

# Load data
results = {}

for name, result in zip(list(MODELS_ours.keys()), results_path):
    with open(result, 'r', encoding='utf-8') as f:
        results[name] = json.load(f)

# Statistics variables for each model
model_stats = {}
for model_name in MODELS.keys():
    model_stats[model_name] = {
        'total_questions': 0,
        'correct_answers': 0,
        'type_stats': {},
        'detailed_results': []
    }

# Iterate through all questions
for name, r in results.items():
    for d in r:
        #question = d['Question:']
        groundtruth = d['Groundtruth:'].strip()
        answer = d['gpt_answer:'].strip()
        question_type = d['question_type']

        stats = model_stats[name]
        stats['total_questions'] += 1
        
        # Initialize type statistics
        if question_type not in stats['type_stats']:
            stats['type_stats'][question_type] = {'correct': 0, 'total': 0}
        
        stats['type_stats'][question_type]['total'] += 1
        
        # Check if correct
        is_correct = groundtruth == answer
        if is_correct:
            stats['correct_answers'] += 1
            stats['type_stats'][question_type]['correct'] += 1
        
        # Save detailed results for further analysis
        stats['detailed_results'].append({
            'question_type': question_type,
            'is_correct': is_correct,
            'groundtruth': groundtruth,
            'model_answer': answer
        })

# for d in results[0]:
#     question = d['Question:']
#     groundtruth = question.get('groundtruth', '').strip()
#     question_type = question.get('question_type', 'unknown')
    
#     # Process each model
#     for model_name, field_name in MODELS.items():
#         model_answer = question.get(field_name, '').strip()
        
#         # Skip if no answer available
#         if not model_answer:
#             continue
            
#         stats = model_stats[model_name]
#         stats['total_questions'] += 1
        
#         # Initialize type statistics
#         if question_type not in stats['type_stats']:
#             stats['type_stats'][question_type] = {'correct': 0, 'total': 0}
        
#         stats['type_stats'][question_type]['total'] += 1
        
#         # Check if correct
#         is_correct = groundtruth == model_answer
#         if is_correct:
#             stats['correct_answers'] += 1
#             stats['type_stats'][question_type]['correct'] += 1
        
#         # Save detailed results for further analysis
#         stats['detailed_results'].append({
#             'question_type': question_type,
#             'is_correct': is_correct,
#             'groundtruth': groundtruth,
#             'model_answer': model_answer
#         })

# Calculate and display results for each model
for model_name in MODELS.keys():
    stats = model_stats[model_name]
    overall_accuracy = (stats['correct_answers'] / stats['total_questions'] * 100) if stats['total_questions'] > 0 else 0
    
    print(f"\n📊 {model_name} Overall Accuracy Statistics:")
    print("=" * 50)
    print(f"Total Questions: {stats['total_questions']}")
    print(f"Correct Answers: {stats['correct_answers']}")
    print(f"Wrong Answers: {stats['total_questions'] - stats['correct_answers']}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    # Calculate accuracy by type
    print(f"\n📋 {model_name} Accuracy by Question Type:")
    type_accuracies = {}
    for question_type, type_stats in stats['type_stats'].items():
        accuracy = (type_stats['correct'] / type_stats['total'] * 100) if type_stats['total'] > 0 else 0
        type_accuracies[question_type] = accuracy
        print(f"  {question_type}: {type_stats['correct']}/{type_stats['total']} ({accuracy:.1f}%)")
    
    stats['type_accuracies'] = type_accuracies
    stats['overall_accuracy'] = overall_accuracy

# 3. Create comparison radar chart
print("\n🎨 Creating Comparison Radar Chart...")

# Ensure all 6 types are included
all_types = ['element_identification', 'axis_and_symmetry', 'element_distribution', 
             'overall_composition', 'preservation_and_restoration', 'material_identification']

# Create figure with subplots
fig = plt.figure(figsize=(24, 8))

# Colors for different models
colors = ['#2E8B57', '#FF6B35', '#9932CC', '#FFD700', '#A36722', "#FF0000", "#0D00FF"]
model_names = list(MODELS.keys())

# Subplot 1: Comparison Radar chart
ax1 = plt.subplot(131, projection='polar')

for i, model_name in enumerate(model_names):
    if model_stats[model_name]['total_questions'] == 0:
        continue
        
    type_accuracies = model_stats[model_name]['type_accuracies']
    values = [type_accuracies.get(qt, 0) for qt in all_types]
    
    # Close the radar chart by connecting first and last points
    values_closed = values + values[:1]
    
    # Calculate angles
    N = len(all_types)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    
    ax1.plot(angles_closed, values_closed, 'o-', linewidth=3, 
             label=f'{model_name}', color=colors[i], markersize=8)
    ax1.fill(angles_closed, values_closed, alpha=0.15, color=colors[i])

# Set labels
categories = [qt.replace('_', ' ').title() for qt in all_types]
ax1.set_xticks(angles)
ax1.set_xticklabels(categories, size=10, weight='bold')
ax1.set_ylim(0, 80)
ax1.set_yticks([10, 20, 30, 40, 50, 60, 70, 80])
ax1.set_yticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%'], size=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Model Comparison by Question Type\n(6 Dimensions)', pad=30, size=14, weight='bold')
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Subplot 2: Bar chart comparison
ax2 = plt.subplot(132)
x = np.arange(len(all_types))
width = 0.1

for i, model_name in enumerate(model_names):
    if model_stats[model_name]['total_questions'] == 0:
        continue
        
    type_accuracies = model_stats[model_name]['type_accuracies']
    values = [type_accuracies.get(qt, 0) for qt in all_types]
    
    bars = ax2.bar(x + i*width, values, width, label=model_name, 
                   color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{value:.1f}%', ha='center', va='bottom', fontsize=9, weight='bold')

ax2.set_xlabel('Question Type', fontsize=12, weight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
ax2.set_title('Model Accuracy Comparison by Type', fontsize=14, weight='bold')
ax2.set_xticks(x + width/2)
ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

# Subplot 3: Overall accuracy comparison
ax3 = plt.subplot(133)
model_names_display = []
overall_accuracies = []

for model_name in model_names:
    if model_stats[model_name]['total_questions'] > 0:
        model_names_display.append(model_name)
        overall_accuracies.append(model_stats[model_name]['overall_accuracy'])

bars = ax3.bar(model_names_display, overall_accuracies, 
               color=colors[:len(model_names_display)], alpha=0.8, 
               edgecolor='black', linewidth=2, width=0.4)

# Add value labels
for bar, value in zip(bars, overall_accuracies):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom', fontsize=14, weight='bold')

ax3.set_ylabel('Overall Accuracy (%)', fontsize=12, weight='bold')
ax3.set_title('Overall Accuracy Comparison', fontsize=14, weight='bold')
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('multi_model_accuracy_comparison_6dimensions.png', bbox_inches='tight', dpi=300)
#plt.show()

# 4. Output detailed comparison table
print("\n📈 Detailed Accuracy Comparison Table:")
comparison_data = []
for qt in all_types:
    row = {'Question Type': qt.replace('_', ' ').title()}
    for model_name in model_names:
        if model_stats[model_name]['total_questions'] > 0:
            accuracy = model_stats[model_name]['type_accuracies'].get(qt, 0)
            row[f'{model_name} (%)'] = f"{accuracy:.1f}%"
        else:
            row[f'{model_name} (%)'] = "N/A"
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# 5. Analysis conclusions
print(f"\n🎯 Comparison Analysis Summary:")
for model_name in model_names:
    stats = model_stats[model_name]
    if stats['total_questions'] == 0:
        print(f"• {model_name}: No data available")
        continue
        
    type_accuracies = stats['type_accuracies']
    if type_accuracies:
        best_type = max(type_accuracies.items(), key=lambda x: x[1])
        worst_type = min(type_accuracies.items(), key=lambda x: x[1])
        
        print(f"• {model_name} Overall Accuracy: {stats['overall_accuracy']:.1f}%")
        print(f"  - Best: {best_type[0].replace('_', ' ').title()} ({best_type[1]:.1f}%)")
        print(f"  - Worst: {worst_type[0].replace('_', ' ').title()} ({worst_type[1]:.1f}%)")

# Performance comparison
available_models = [m for m in model_names if model_stats[m]['total_questions'] > 0]
if len(available_models) >= 2:
    model1, model2 = available_models[0], available_models[1]
    acc1 = model_stats[model1]['overall_accuracy']
    acc2 = model_stats[model2]['overall_accuracy']
    
    if acc1 > acc2:
        print(f"• Winner: {model1} outperforms {model2} by {acc1-acc2:.1f} percentage points")
    elif acc2 > acc1:
        print(f"• Winner: {model2} outperforms {model1} by {acc2-acc1:.1f} percentage points")
    else:
        print(f"• Result: {model1} and {model2} have equal performance") 