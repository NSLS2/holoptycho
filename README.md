# holoscan-framework

This repo contains the implementation for real-time ptychography reconstruction using Holoscan framework. Below are the instructions on deploying the code.


## Prerequisites
Both `ptycho_gui` and `ptycho` repos should be cloned to a folder placed one level above the folder containing the current repo and set to the correct development branches (they get periodically updated), e.g.:

```
<project folder>
    |---/ptycho/                [branch: holoscan]
    |---/ptycho_gui/            [branch: HXN_update_2024Q3]
    |---/holoscan-framework/    [branch: main]
```

## Holoscan App Container
The code for the Holoscan application is contained in the folder `eiger_dir/`. The Holoscan application can be launched by running the `eiger_connect_sample.py` script.

In order to run, we need to build a container defined in the [Dockerfile](https://github.com/skarakuzu/holoscan-framework/blob/ptycho_step_wise_array_update/eiger_dir/Dockerfile).

To build a container named `hxn-ptycho-holoscan`:
```
docker build ./eiger_dir -t hxn-ptycho-holoscan --network host
```

If docker/podman complains about groupid, try:
```
podman build --userns-uid-map=0:0:1 --userns-uid-map=1:1:1999 --userns-uid-map=65534:2000:2 ./eiger_dir -t hxn-ptycho-holoscan --network host
```

To turn the Docker container into a podman container, we run the following command:
```
podman pull docker-daemon:hxn-ptycho-holoscan:latest
```
You can now verify that podman sees the correct image via:
```
podman image ls
```
The output should look something like this:
```
REPOSITORY                               TAG                      IMAGE ID      CREATED        SIZE
docker.io/library/hxn-ptycho-holoscan    latest                   9777387459f9  22 hours ago   411 MB

```

After successfully building the container, we run it via
```
podman run --rm --net host -it --privileged\
    -v ./eiger_dir:/eiger_dir \
    -v ./eiger_simulation/test_data:/test_data \
    -v ../ptycho_gui:/ptycho_gui \
    -v ../ptycho:/ptycho_gui/nsls2ptycho/core/ptycho \
    -w /eiger_dir \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 \
    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    -e OMPI_COMM_WORLD_LOCAL_RANK=0 \
    -e OMPI_COMM_WORLD_LOCAL_SIZE=1 \
    -e HOLOSCAN_ENABLE_PROFILE=1 \
    --device nvidia.com/gpu=all hxn-ptycho-holoscan
```

Optional with docker (should be launched from `project folder` containing `holoscan-framework`):
```
docker run --rm --net host -it --privileged --ipc=host --runtime=nvidia --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ./holoscan-framework/eiger_dir:/eiger_dir \
    -v ./holoscan-framework/eiger_simulation/test_data:/test_data \
    -v ./ptycho_gui:/ptycho_gui \
    -v ./ptycho:/ptycho_gui/nsls2ptycho/core/ptycho \
    -w /eiger_dir \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 \
    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    -e OMPI_COMM_WORLD_LOCAL_RANK=0 \
    -e OMPI_COMM_WORLD_LOCAL_SIZE=1 \
    -e HOLOSCAN_ENABLE_PROFILE=1 \
    hxn-ptycho-holoscan
```

Note that the directories for `ptycho_gui` and `ptycho` are mounted inside the container.

Since it is easier to manage virtual environment, packaging and version control via `pixi`, we use the following `pixi.toml` to generate a virtual conda environment inside a directory named `eiger_holoscan` we mounted while starting the container

To install the environment run the following commands:
```
pixi install
```
This command will start executing the `pixi.toml` file to create the virtual environment and create a file named `pixi.lock` which shows the information about installed packages. If one needs to remove the existing environment to start a fresh one, it is possible to do it via the `pixi clean` command.

Installation of ptycho code is done separately using pixi command "postinstall" configured in `pixi.toml` file. To install ptycho code environment run the following:
```
pixi run postinstall
```

If ptycho is installed on the machine for the first time, navigate, within the container, to `/ptycho_gui/nsls2ptycho/core/ptycho` and run `configure.sh` script to generate `.cubin` files.

To enable the pixi environment run
```
pixi shell
```

To run the full holoscan example, run:
```
python3 pipeline_ptycho.py
```

Alternatively, only the Rx part of the pipeline can be run by executing
```
python3 pipeline_source.py
```

To run Rx and the preprocessor operators, execute
```
python3 pipeline_preprocess.py
```

To test the holoscan pipeline in different environments, one can modify the `holoscan_config.yaml` file, or create another one, e.g. `holoscan_test_config.yaml`. A new config then must be passed to the script as an argument:
```
python3 pipeline_ptycho.py --config holoscan_test_config.yaml
```

## Profiling the pipeline with nsight systems

When profiling the pipeline, the Linux operating system’s perf_event_paranoid level must be 2 or less. Use the following command to check:
```
cat /proc/sys/kernel/perf_event_paranoid
```
If the output is >2, then do the following to temporarily adjust the paranoid level (note that this has to be done after each reboot):
```
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
```

Use the following command to profile the pipeline with simulated data stream (add `--config holoscan_test_config.yaml` at the end if neccessary):
```
nsys profile -t cuda,nvtx,osrt,python-gil -o ptycho_profile.nsys-rep -f true -d 30 python3 pipeline_ptycho.py
```



## Simulating data stream using test data from HXN
To test/develop the holoscan pipeline, we can run a simulated data stream.
Test ptychography scan data recoreded by Eiger should be placed to `/eiger_simulation/test_data/`. Currently, the test files include `scan_257331_raw.h5` and `scan_257331.h5`. See communications with Zirui to get access to these files.

To emulate the Eiger data stream, a simulated SimplonAPI 1.8 is used.

To build the container for simulated API:

```
docker build ./eiger_simulation -t eiger_sim:test --network host
```

The API uses ports 8000 and 5555 for the simulated detector control and data stream, respectively.

To run this container with podman, first pull it:
```
podman pull docker-daemon:eiger_sim:test
```
Once it's done, check the availability of the container:
```
podman image ls

# output:
REPOSITORY                               TAG                      IMAGE ID      CREATED        SIZE
docker.io/library/eiger_sim              test                     da9a38ed0b93  2 weeks ago    2.35 GB
```


To see the API output run the container interactively:

```
# with podman:
podman run -it -p 8000:8000 -p 5555:5555 eiger_sim:test
# with docker:
docker run -it -p 8000:8000 -p 5555:5555 eiger_sim:test
```


Otherwise, run it in the detached mode:

```
# with podman:
podman run -d -p 8000:8000 -p 5555:5555 eiger_sim:test

# with docker:
docker run -d -p 8000:8000 -p 5555:5555 eiger_sim:test
```

After launching the container (if container is running interactively, open a separate terminal), find the container ID with `podman ps` (or `docker ps`) command. The output should look like this:
```
CONTAINER ID   IMAGE            COMMAND                  CREATED              STATUS              PORTS                                                                                  NAMES
d270120da233   docker.io/library/eiger_sim:test   "/bin/sh -c 'uvicorn…"   About a minute ago   Up About a minute   0.0.0.0:5555->5555/tcp, :::5555->5555/tcp, 0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   peaceful_meitner
```
Connect to the container:
```
# with podman:
podman exec -it d270120da233 bash

# with docker:
docker exec -it d270120da233 bash
```
To trigger the detector use the following command:
```
python trigger_detector.py -n 10000 -dt 0.001
```
parameter `-n` controls how many images will be transmitted by the API. Once executed, you will see the frame sending status in the API window (if it is open in the interactive mode). The holoscan application window will show frame receiving status.



## Docker instructions (optional)


Note: to run docker comands, one needs to be in `<project folder>`, i.e. outside of `/holoscan-framework/`

without viz:



Proceed with installing pixi environment as described above.
