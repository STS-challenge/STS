# 📋 Docker Instruction
First, make sure that `Docker` and `NVIDIA Container Toolkit` are installed on your computing platform as they are essential for the algorithm packaging.
The former ensures that you can perform the packaging, while the latter enables GPU utilization within Docker.
Be sure to confirm that your system has been properly installed and configured.

Next, make sure to modify the `requirements.txt` file to include the necessary dependencies for your code project.
This ensures that all the required libraries for the prediction process are included so that the prediction code can be executed correctly and produce the desired results.

Then,
implement your inference testing process in the `run_inference.py` file.
After that, execute the `build.sh` script to troubleshoot any errors. 


Finally, proceed to execute the `export.sh` script to export the `STS2024_Algorithm_Docker.tar.gz` file that can be sent to our challenge email.
