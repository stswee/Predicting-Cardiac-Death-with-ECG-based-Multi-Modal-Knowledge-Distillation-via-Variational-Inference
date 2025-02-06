## Instructions on Dataset Download

1. Open 2+ **cpu-only** nodes for fastest download speeds

    salloc --partition=cpu --mem=4G --ntasks=1 --cpus-per-task=1 --account=bdlo-delta-cpu --time=01:00:00

2. Run <code>bash INSTALL_DATA_n1_partial_download.sh</code> on one of the nodes. This file goes through the ECG data and resumes any partial downloads, in addition to downloading new files.

3. Run <code>bash INSTALL_DATA_n2.sh</code> on the rest of the nodes. This file skips partial downloads and just downloads new files. 


**Remarks**

 - Only 1 cpu/task should be used to save hours on cluster :(
 - It should download a ~100mb file in 2 min. If it doesnt ur doing something wrong 