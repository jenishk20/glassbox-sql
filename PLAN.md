# Plan

The working plan for the Glass Box series, kept in the open. Each day has a project task and a maths
task; the maths builds from the basics to DPO (week 1) and GRPO (week 2).

## Weeks 1 and 2 (21 Oct to 3 Nov 2026)

### Week 1: first results, Part 1, DPO

- [ ] **Wed 21 Oct**
  - [ ] Project: push the code; one-time cluster setup (clone, container, model weights, BIRD)
  - [ ] Maths: logits, softmax, temperature, greedy decoding
- [ ] **Thu 22 Oct**
  - [ ] Project: `sbatch scripts/slurm/check.sbatch` passes; submit `part2_qwen.sbatch` overnight
  - [ ] Maths: log-likelihood and cross-entropy (the SFT loss); score a sentence's log-probability with a small Qwen
- [ ] **Fri 23 Oct**
  - [ ] Project: copy results back; read the report and 25 failure traces each for Base and Instruct
  - [ ] Maths: KL divergence (definition, asymmetry, why training stays close to a reference model)
- [ ] **Sat 24 Oct**
  - [ ] Project: W&B runs (5-question test first, then full runs for 4 open models)
  - [ ] Maths: Bradley-Terry model (turning "A is better than B" into a probability)
- [ ] **Sun 25 Oct**
  - [ ] Project: draft Part 1; add KaTeX to the blog
  - [ ] Maths: confidence intervals and the McNemar test, by hand
- [ ] **Mon 26 Oct**
  - [ ] Project: review the Part 1 draft; report dev-1106 question 178 (`julianday('now')`) to the BIRD team
  - [ ] Maths: derive the DPO loss from the RLHF objective ([arXiv 2305.18290](https://arxiv.org/abs/2305.18290), section 4 and appendix A)
- [ ] **Tue 27 Oct**
  - [ ] Project: publish Part 1 and an X thread
  - [ ] Maths: implement the DPO loss in about 20 lines; check it against TRL's on a toy batch

### Week 2: Part 2, GRPO

- [ ] **Wed 28 Oct**
  - [ ] Project: draft Part 2 (Base vs Instruct vs open models, failure analysis)
  - [ ] Maths: policy gradient, the log-derivative trick, REINFORCE
- [ ] **Thu 29 Oct**
  - [ ] Project: upload the Part 2 run folders as a Hugging Face dataset (CC BY-SA 4.0)
  - [ ] Maths: baselines and variance (why subtracting a baseline helps)
- [ ] **Fri 30 Oct**
  - [ ] Project: publish Part 2 and an X thread
  - [ ] Maths: PPO's probability ratio and clipping ([arXiv 1707.06347](https://arxiv.org/abs/1707.06347))
- [ ] **Sat 31 Oct**
  - [ ] Project: catch-up
  - [ ] Maths: GRPO, the group-relative advantage, no critic, the KL term ([arXiv 2402.03300](https://arxiv.org/abs/2402.03300), section 4)
- [ ] **Sun 1 Nov**
  - [ ] Project: submit the Part 3 prompt-ladder job overnight
  - [ ] Maths: GRPO by hand on 8 answers with 0/1 rewards; show that a fraction 1 − p^G − (1−p)^G of groups carry signal
- [ ] **Mon 2 Nov**
  - [ ] Project: read the Part 3 results
  - [ ] Maths: write up DPO and GRPO from scratch (drafts for Parts 8 and 9)
- [ ] **Tue 3 Nov**: buffer

## After that

| Weeks | Part |
|---|---|
| 3 | Part 3: what each prompt ingredient is worth |
| 4 | Part 4: frontier models and cost per correct answer |
| 5 to 6 | Part 5: supervised fine-tuning |
| 7 | Part 6: distilling reasoning |
| 8 | Part 7: voting and pass@k |
| 9 | Submit the best model so far to the BIRD hidden test set (results take up to 10 days) |
| 9 to 10 | Part 8: DPO |
| 11 to 12 | Part 9: GRPO |
| 13 to 14 | Part 10: final results |
