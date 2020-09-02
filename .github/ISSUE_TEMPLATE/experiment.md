---
name: Experiment
about: Report experimental results under Issues
title: "[M.m] Name of Experiment"
labels: experiment
assignees: andrewjong, gauravkuppa, veralauee

---

<!--- Just fill in the major+minor version number where [M.m] is in the title. Patch will be reported in comments. -->

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
# Experiment Description
<!--- 
For Experiment Number, use "Major.minor.patch", e.g. 1.2.0.
Major.minor should match the [M.m] in the title. 
Patch describes a bug fix (change in the code or branch).
-->
**Experiment Number:** 1.2.0
**Branch:** `master`
**Timestamp:** MM/DD/YYYY 9pm PT

# Loss Graphs
<!--- Put detailed loss graphs here. Please include all graphs! -->

# Image Results
<!--- Put detailed image results here. Please include all images! Multiple screenshots is good. -->

# Comments, Observations, or Insights
<!--- Optional -->

```
