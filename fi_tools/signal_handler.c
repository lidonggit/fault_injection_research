void show_stackframe() {
  void *trace[16];
  char **messages = (char **)NULL;
  int i, trace_size = 0;

  trace_size = backtrace(trace, 16);
  messages = backtrace_symbols(trace, trace_size);
  printf("[bt] Execution path:\n");
  for (i=0; i<trace_size; ++i)
	printf("[bt] %s\n", messages[i]);
}

void sig_func(int sig)
{
 write(1, "Caught signal from fault injector ... \n", 17);
 signal(1,sig_func);
 show_stackframe();
}


