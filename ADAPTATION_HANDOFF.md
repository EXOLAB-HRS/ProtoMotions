# preview13 adaptation source preservation

This merge retains both hc_jy and preview13 histories. For the four overlapping simulator/recording files, the tree uses exactly the executed preview13 source (including native_substep_observer hooks), rather than mixing untested legacy camera settings into the physics path. The earlier hc_jy camera configuration remains available at commit 7b2cc444045f2920e36eaa0775242b9749e1b74c. No physics was rerun for this merge.
