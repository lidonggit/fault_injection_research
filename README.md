fault_injection_research
========================
<< benchmarks >>
all benchmarks

<< experiments >>
fault injection results


<< fi_tools >>
There is a very lightweight fault injection tool in this directory. 
This tool basically creates a helper thread to 
perform fault injection. To use the tool, a specific API, "launch_fi_thread(void* start_address, int mem_size)", 
must be inserted into the application. In particular, for a global data structure, 
the API can be inserted right before the main computation happens;
for a heap or stack data structure, the API can be inserted right before the target 
data structure is allocated.

To perform fault injection, the user must input three parameters in a configuration file.
These three parameters are, time randomness parameter (TRP), space randomness parameter (SRP),
and execution time.

TRP and SRP should be in [0,1]. The execution time refers to the major computation time (for global data)
or the data liveness time (for heap or stack data). Given an execution time "t" and a data size "s",
a fault of one bit-flip will be performed at the time point (t * TRP) and at the location (s * SRP).
For example, given a data structure with t=10s, s=256KB, TRP=0.12, and SRP=0.45, a fault will be injected
at 1.2s and at 256*1024*8*0.45 bit

By default, the configuration file is named as "config_file.cfg" in the application directory.
But the user can specify a specific file location by setting a environment variable "FI_CONFIG_PATH".
For example in BASH, "export FI_CONFIG_PATH=/home/dd/work/S3D/config_file.cfg"
