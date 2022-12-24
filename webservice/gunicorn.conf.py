from multiprocessing import cpu_count


bind = ["0.0.0.0:8000"]

loglevel = "info"

workers = cpu_count()*2 + 1
