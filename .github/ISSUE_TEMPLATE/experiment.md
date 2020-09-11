---
name: Experiment
about: Report experimental results under Issues
title: "[M.m] Name of Experiment"
labels: experiment
assignees: andrewjong, gauravkuppa, veralauee

---

<!--- Fill in the Major.minor version number where [M.m] is in the title. 
- Major is an experiment category. Closely related experiments are grouped under the same major version.
- Minor is a small change in the train command, e.g. a different hyperparameter value.
- Patch describes a bug fix (change in the code or branch), or rerun.
- Separate issues should be created for each [M.m]. Patches are reported in comments. -->

# Description
Explain why we're running this and what we expect.

**Planned Start Date:**

**Depends on Previous Experiment?**  Y/N

# Train Command
```bash
python train.py ___
```

# Report Results
To report a result, copy this into a comment below:
```
# Result Description
<!--- 
For Experiment Number, use "Major.minor.patch", e.g. 1.2.0.
Major.minor should match the [M.m] in the title. 
Patch describes a bug fix (change in the code or branch).
-->
**Experiment Number:** 1.2.0
**Branch:** `master`
**Timestamp:** MM/DD/YYYY 9pm PT
**Epochs:** 


# Architecture
**Model Layers:**
<!-- Paste the printed Model Layers -->

**Module Parameters:**
<!-- Paste the Params table -->


# Loss Graphs
<!--- Put detailed loss graphs here. Please include all graphs! -->

# Image Results
<!--- Put detailed image results here. Please include all images! Multiple screenshots is good. -->

# Comments, Observations, or Insights
<!--- Optional -->
```

- [ ] Open GitHub Issue
- [ ] Start training with tmux (tensorboard and training)
- [ ] Upload scalars, train, and validation images to GitHub
- [ ] Upload checkpoints to Google Drive
- [ ] Generate test results from latest epoch
- [ ] Calculate metrics (PSNR, SSIM)
- [ ] Visualize metrics
