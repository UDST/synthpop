# coding: utf-8
from synthpop.recipes.starter2 import Starter
from synthpop.synthesizer import synthesize_all, enable_logging
import pandas as pd
import os
import sys
import numpy as np
from multiprocessing import Process
import uuid
from multiprocessing import Manager


def run_all(ns, offset):
    index_to_process = ns.jobs_per_process[offset]
    print("Process[%d] Got %d indexes" % (os.getpid(), len(index_to_process)))

    indexes = []
    for item in index_to_process:
        indexes.append(pd.Series(item, index=["state", "county", "tract", "block group"]))

    starter = Starter(os.environ["CENSUS"], ns.state_abbr, ns.county_name)

    households, people, fit_quality = synthesize_all(starter, indexes=indexes)

    hh_file_name = "household_{}_{}_{}.csv".format(ns.state_abbr, ns.county_name, offset)
    people_file_name = "people_{}_{}_{}.csv".format(ns.state_abbr, ns.county_name, offset)

    households.to_csv(hh_file_name, index=None, header=True)
    people.to_csv(people_file_name, index=None, header=True)

    for geo, qual in fit_quality.items():
        print ('Geography: {} {} {} {}'.format(
            geo.state, geo.county, geo.tract, geo.block_group))
        # print '    household chisq: {}'.format(qual.household_chisq)
        # print '    household p:     {}'.format(qual.household_p)
        print ('    people chisq:    {}'.format(qual.people_chisq))
        print ('    people p:        {}'.format(qual.people_p))


if __name__ == "__main__":
    state_abbr = sys.argv[1]
    county_name = sys.argv[2]
    workers = int(os.cpu_count() / 2)
    if 'sched_getaffinity' in dir(os):
        workers = int(len(os.sched_getaffinity(0)) / 2)
    if os.environ.get("N_WORKERS") is not None:
        workers = int(os.environ["N_WORKERS"])
    if len(sys.argv) > 3:
        state, county, tract, block_group = sys.argv[3:]
        indexes = [pd.Series(
            [state, county, tract, block_group],
            index=["state", "county", "tract", "block group"])]
    else:
        indexes = None

    starter = Starter(os.environ["CENSUS"], state_abbr, county_name)
    enable_logging()

    if indexes is None:
        indexes = list(starter.get_available_geography_ids())
    if len(indexes) < workers:
        workers = len(indexes)

    print("Workers: %d" % (workers))
    print("Indexes: %d" % (len(indexes)))

    jobs_per_process = np.array_split(indexes, workers)
    mgr = Manager()
    ns = mgr.Namespace()
    ns.jobs_per_process = jobs_per_process
    ns.state_abbr = state_abbr
    ns.county_name = county_name

    processes = []
    for i in range(0, len(jobs_per_process)):
        p = Process(target=run_all, args=(ns, i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        print("Process %d exit code is: %d" % (p.pid, p.exitcode))
        if p.exitcode != 0:
            print("Process %d has exited unexpectedly, the results are not correct or full!"  % (p.pid))
