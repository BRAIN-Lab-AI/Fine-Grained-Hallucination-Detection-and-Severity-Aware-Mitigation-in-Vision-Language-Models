# Reference Audit

This audit records the citation source used for each BibTeX entry in `references.bib`, the claim it supports in the hallucination mitigation paper, and any remaining metadata uncertainty.

| Citation key | Verified source | Supports | Metadata notes |
| --- | --- | --- | --- |
| `chen2024unified` | ACL Anthology: <https://aclanthology.org/2024.acl-long.178> | Unified multimodal hallucination detection, MHaluBench, UNIHD. | Template omitted Lei Liang and used arXiv-only metadata. ACL 2024 venue, DOI, and pages were verified from ACL Anthology. |
| `gunjal2024detecting` | AAAI proceedings: <https://ojs.aaai.org/index.php/AAAI/article/view/29771> | Fine-grained hallucination detection/prevention, M-HalDetect, FDPO, reward-model/rejection-sampling mitigation. | AAAI volume, issue, pages, DOI, and authors verified. |
| `li2023silkie` | arXiv: <https://arxiv.org/abs/2312.10665> | Preference distillation, VLFeedback, DPO-based LVLM alignment, reduced hallucination. | No peer-reviewed venue was verified; kept as arXiv preprint. |
| `liu2024survey` | arXiv: <https://arxiv.org/abs/2402.00253> | Background taxonomy and survey coverage for LVLM hallucination symptoms, causes, evaluation, and mitigation. | No peer-reviewed venue was verified; kept as arXiv preprint. |
| `li2023blip2` | PMLR/ICML 2023: <https://proceedings.mlr.press/v202/li23q.html> | Background on bridging frozen image encoders and frozen LLMs for efficient vision-language pretraining. | ICML 2023 venue, PMLR volume, pages, and authors verified from PMLR. |
| `liu2023visual` | NeurIPS proceedings: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html> | LLaVA and multimodal visual instruction tuning background. | NeurIPS 2023 venue and authors verified from proceedings. |
| `dai2023instructblip` | NeurIPS proceedings: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/9a6a435e75419a836fe47ab6793623e6-Abstract-Conference.html> | Instruction-aware visual feature extraction and vision-language instruction tuning background. | NeurIPS 2023 venue and authors verified from proceedings. |
| `liu2024robust` | OpenReview ICLR 2024: <https://openreview.net/forum?id=J44HfH4JCg> | Robust instruction tuning and LRV-Instruction as a training-based mitigation method. | Published ICLR 2024 poster verified. |
| `rafailov2024dpo` | NeurIPS proceedings: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html> | DPO objective used as the foundation for preference optimization. | The existing key says 2024, but NeurIPS lists the paper in NeurIPS 2023/volume 36. The BibTeX keeps the existing key for compatibility and uses `year = {2023}`. |
| `rohrbach2018object` | ACL Anthology: <https://aclanthology.org/D18-1437> | Object hallucination in image captioning and CHAIR-style object hallucination evaluation. | Template used arXiv-only metadata. EMNLP 2018 venue, DOI, and pages were verified. |
| `xiao2025hsa` | AAAI proceedings: <https://ojs.aaai.org/index.php/AAAI/article/view/34744> | Fine-grained AI feedback, detect-then-rewrite preference construction, hallucination severity-aware DPO. | AAAI page lists Fangxun Shu as the seventh author. Some pre-publication metadata lists different authors; the BibTeX follows the AAAI proceedings version. |
| `yu2024rlhfv` | CVF Open Access: <https://openaccess.thecvf.com/content/CVPR2024/html/Yu_RLHF-V_Towards_Trustworthy_MLLMs_via_Behavior_Alignment_from_Fine-grained_Correctional_CVPR_2024_paper.html> | Fine-grained correctional human feedback and dense DPO for trustworthy MLLMs. | CVPR 2024 pages and author list verified from CVF. |
| `zhou2024analyzing` | OpenReview ICLR 2024: <https://openreview.net/forum?id=oZDJKTlOUe> | Analysis of object hallucination factors and training-free LURE mitigation. | Published ICLR 2024 poster verified. |
| `li2023pope` | ACL Anthology: <https://aclanthology.org/2023.emnlp-main.20> | POPE benchmark/polling-based object hallucination evaluation. | EMNLP 2023 venue, DOI, pages, and authors verified. Use this key for POPE. |
| `wang2023amber` | arXiv: <https://arxiv.org/abs/2311.07397> | AMBER benchmark for LLM-free multidimensional MLLM hallucination evaluation. | No peer-reviewed venue was verified; kept as arXiv preprint. Use this key for AMBER. |
| `guan2024hallusionbench` | CVF Open Access: <https://openaccess.thecvf.com/content/CVPR2024/html/Guan_HallusionBench_An_Advanced_Diagnostic_Suite_for_Entangled_Language_Hallucination_and_CVPR_2024_paper.html> | HallusionBench diagnostic benchmark for entangled language hallucination and visual illusion. | CVPR 2024 venue, pages, and author list verified from CVF. |
| `sun2024factrlhf` | ACL Anthology: <https://aclanthology.org/2024.findings-acl.775> | Factually augmented RLHF, LLaVA-RLHF, and MMHal-Bench evaluation context. | Added as a useful optional benchmark/alignment citation. Findings ACL 2024 venue, DOI, pages, and authors verified. |

## Benchmark Citation Guidance

- Use `li2023pope` when discussing POPE or polling-based object probing evaluation.
- Use `wang2023amber` when discussing AMBER.
- Use `guan2024hallusionbench` when discussing image-context reasoning diagnostics, language hallucination, and visual illusion.
- Use `li2023blip2`, `liu2023visual`, and `dai2023instructblip` when introducing modern LVLM architecture and instruction-tuning context.
- Use `rohrbach2018object` for CHAIR and object hallucination in image captioning.
- Treat Object HalBench as not having a clearly separate primary citation based on this audit. In current LVLM papers it is commonly tied back to object hallucination/CHAIR-style evaluation; cite `rohrbach2018object` for the benchmark basis and `xiao2025hsa` when referencing the specific Object HalBench results reported by HSA-DPO.
- Use `sun2024factrlhf` only if the paper discusses MMHal-Bench or LLaVA-RLHF/Fact-RLHF directly.
