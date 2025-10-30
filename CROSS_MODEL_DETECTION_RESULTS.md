# Cross-Model Detection: GPT-5 Identifying Claude's Code

## Executive Summary

We tested whether GPT-5 can identify code written by Claude (Anthropic), and discovered a **surprising result**: GPT-5 is significantly better at identifying Claude's code than identifying its own code!

## Key Results

| Task | Accuracy | Details |
|------|----------|----------|
| **Self-Recognition** | 79.2% | GPT-5 identifying its own code (202/255 correct) |
| **Cross-Model Detection** | **97.7%** | GPT-5 identifying Claude's code (251/257 correct) |

**Improvement: +18.5 percentage points**

## Experimental Setup

### Self-Recognition Baseline
- **Judge**: GPT-5
- **Task**: Given two code snippets (one from GPT-5, one from Claude), identify which one GPT-5 wrote
- **Dataset**: MBPP-sanitized test set
- **Samples**: 255 pairs

### Cross-Model Detection
- **Judge**: GPT-5 (fixed monitor)
- **Target**: Claude (anthropic/claude-haiku-4.5)
- **Task**: Given two code snippets (one from Claude, one from another model), identify which one Claude wrote
- **Dataset**: MBPP-sanitized test set
- **Samples**: 257 pairs

## Interpretation

This surprising finding suggests several important insights:

1. **Distinctive Model Signatures**: Claude has very distinctive coding patterns that are easier for GPT-5 to identify compared to GPT-5's own patterns.

2. **Cross-Family Detection is Easier**: It may be easier to detect code from different model families than to recognize one's own output. This suggests that different families (Anthropic vs OpenAI) have more distinguishable characteristics.

3. **Monitoring Potential**: Strong models like GPT-5 can act as effective "monitors" to detect code from other LLM families with high accuracy.

## Implications for Research

### Updated Motivation
The paper can now argue that:
- Stronger models like GPT-5 can identify code written by other LLM families with very high accuracy
- Cross-model detection could be a viable approach for AI output monitoring and verification
- This has implications for AI safety, attribution, and detecting AI-generated content

### Potential Observable Features
What makes Claude's code so identifiable? Possible distinctive features:
- **Documentation style**: Claude is known for adding extensive docstrings and comments
- **Variable naming conventions**: Different patterns in naming
- **Code structure**: How functions/classes are organized
- **Error handling patterns**: Different approaches to edge cases
- **Pythonic idioms**: Different preferences for list comprehensions, generators, etc.

## Files Generated

1. **Script**: `cross_model_detection.py` - New script for cross-model detection experiments
2. **Prompt**: `prompts/cross_model_detection.md` - Prompt template for asking judge to identify target model's code
3. **Results**: `data/cross_model_detection/mbpp-sanitized/test/judge-openai-gpt-5_target-anthropic-claude-haiku-4.5.jsonl`
4. **Analysis**: `notebooks/cross_model_detection_analysis.ipynb` - Detailed analysis notebook

## How to Run

```bash
# Test GPT-5's ability to identify Claude's code
poetry run python cross_model_detection.py \
  --dataset-folder mbpp-sanitized \
  --split test \
  --judge openai/gpt-5 \
  --target anthropic/claude-haiku-4.5

# Test other combinations
poetry run python cross_model_detection.py \
  --dataset-folder mbpp-sanitized \
  --split test \
  --judge openai/gpt-5 \
  --target google/gemini-2.5-flash
```

## Next Steps

1. **Test other judge-target combinations**:
   - Claude detecting GPT-5
   - GPT-5 detecting other models (Gemini, DeepSeek, Grok)
   - Claude detecting other models

2. **Feature analysis**:
   - Analyze what specific code features make Claude identifiable
   - Look at examples where GPT-5 made errors

3. **Cross-dataset validation**:
   - Test on other datasets beyond MBPP
   - Test on different programming languages

4. **Fine-tuning experiments**:
   - Can we improve detection accuracy with fine-tuning?
   - What's the minimum training data needed?

## Conclusion

The 97.7% accuracy of GPT-5 in detecting Claude's code is a strong result that can significantly strengthen the paper's motivation. It demonstrates that cross-model detection is not only feasible but highly effective, opening up new avenues for AI monitoring and attribution research.

