# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import six
import uuid
import copy
import logging
import signal
import threading
import time
from .transformer import ProxiedDataset

logger = logging.getLogger(__name__)


class EndSignal(object):
    def __init__(self, errno=0, errmsg=''):
        self.errno = errno
        self.errmsg = errmsg


class ParallelizedDataset(ProxiedDataset):
    """
    Transform samples parallely using multiple workers (threads or processes)

    Notes:
        this class is not thread-safe
    """
    def __init__(self, source_builder, worker_args):
        super(ParallelizedDataset, self).__init__(source_builder.get_input())
        args = {'bufsize': 32, 'worker_num': 8,
            'use_process': True, 'memsize': '3G'}
        if worker_args is not None:
            for k, v in worker_args.items():
                if v is not None:
                    args[k] = v

        if type(args['memsize']) is str:
            assert args['memsize'][-1] == 'G', \
                "invalid param for memsize[%s]" % (args['memsize'])
            gb = args['memsize'][:-1]
            args['memsize'] = int(gb) * 1024 ** 3

        self._worker_args = args
        self._source_builder = source_builder
        self._stopped = False
        self._setup()

    def _setup(self):
        """setup output queue and workers """
        use_process = False
        if 'use_process' in self._worker_args:
            use_process = self._worker_args['use_process']

        bufsize = self._worker_args['bufsize']
        if use_process:
            from .shared_queue import SharedQueue as Queue
            from multiprocessing import Process as Worker
            from multiprocessing import Event
            memsize = int(self._worker_args['memsize'])
            self._outq = Queue(bufsize, memsize=memsize)
        else:
            if six.PY3:
                from queue import Queue
            else:
                from Queue import Queue
            from threading import Thread as Worker
            from threading import Event
            self._outq = Queue(bufsize)

        producer_num = self._worker_args['worker_num']
        self._producers = []
        id = str(uuid.uuid4())[-3:]
        # make input source ready for copy
        self._source_builder.get_input().reset()
        for i in range(producer_num):
            builder = self._source_builder
            p = Worker(
                target=self._produce,
                args=('producer-' + id + '_' + str(i),
                        builder, i, producer_num, self._outq))
            self._producers.append(p)
            p.daemon = True

        self._epoch = -1
        self._feed_ev = Event()
        self._feed_ev.clear()
        self._exit_ev = Event()
        self._exit_ev.clear()
        self._paused_workers = 0
        self._stopped_workers = 0

    def _produce(self, id, builder, part_id, part_num, outq):
        """Fetch data from 'source', process it and put result to 'outq'"""
        builder._parallel_conf = None
        source = builder.build(
            part_id=part_id, part_num=part_num)
        epoch = 0
        self._feed_ev.wait()
        msg = 'worker[%s]' % (id)
        epoch_start = time.time()
        epoch_interval = 1.0
        while True:
            if self._exit_ev.is_set():
                msg = "worker[%s] received exit event, so exit" % (id)
                break

            try:
                sample = source.next()
                outq.put(sample)
            except StopIteration as e:
                epoch += 1
                end = EndSignal(epoch, "worker[%s] finished epoch %d "
                    "partition[%d]" % (id, epoch, part_id))
                if time.time() - epoch_start < epoch_interval:
                    time.sleep(epoch_interval)
                self._feed_ev.clear()
                outq.put(end)
                self._feed_ev.wait()
                source.reset()
                epoch_start = time.time()
            except Exception as e:
                msg = "worker[%s] failed to fetch next sample " \
                    "with error: %s" % (id, str(e))
                logger.warn(msg)
                break
        outq.put(EndSignal(-1, msg))

    def drained(self):
        """ is all data drained for this epoch
        """
        if self._stopped:
            return True

        assert self._epoch >= 0, "first epoch has not started yet"
        no_active_workers = self._paused_workers + self._stopped_workers
        return no_active_workers == len(self._producers)

    def stop(self):
        """ notify to exit
        """
        self._stopped = True
        self._exit_ev.set()
        self._feed_ev.set()

    def next(self):
        """ get next sample from result queue
        """
        if self._epoch < 0:
            self.reset()

        if self.drained():
            raise StopIteration()

        while True:
            sample = self._outq.get()
            if isinstance(sample, EndSignal):
                if sample.errno < 0:
                    logger.warn("producer failed with error: %s" % (sample.errmsg))
                    self._stopped_workers += 1
                elif sample.errno > 0:
                    logger.debug("producer paused with msg: %s" % (sample.errmsg))
                    self._paused_workers += 1

                if self.drained():
                    raise StopIteration()
            else:
                return sample

    def reset(self):
        """ reset for a new epoch of samples
        """
        if self._epoch < 0:
            self._epoch = 0
            for p in self._producers:
                p.start()
        else:
            if not self.drained():
                logger.warn("some worker is still working, "
                    "not all data have been consummed!")
            assert self._paused_workers == len(self._producers),\
                "not all worker finish previouse epoch"
            self._paused_workers = 0
            self._epoch += 1

        stoped_num = self._stopped_workers
        assert stoped_num == 0, "%d workers are stopped, "\
            "cannot launch next epoch[%d]" % (stoped_num, self._epoch)

        self._feed_ev.set()

