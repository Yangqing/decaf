from decaf.scripts import jeffnet
from decaf import util
from decaf.util import smalldata
import numpy as np
import cProfile as profile

# We will use a larger figure size since many figures are fairly big.
data_root='/u/vis/common/deeplearning/models/'
net = jeffnet.JeffNet(data_root+'imagenet.jeffnet.epoch90', data_root+'imagenet.jeffnet.meta')
lena = smalldata.lena()
# run a pass to initialize data
scores = net.classify(lena)
timer = util.Timer()

print 'Testing single classification with 10-part voting (10 runs)...'
timer.reset()
for i in range(10):
    scores = net.classify(lena)
print 'Elapsed %s' % timer.total()

print 'Testing single classification with center_only (10 runs)...'
timer.reset()
for i in range(10):
    scores = net.classify(lena, center_only=True)
print 'Elapsed %s' % timer.total()

lena_ready = lena[np.newaxis, :227,:227].astype(np.float32)
print 'Testing direct classification (10 runs)...'
timer.reset()
for i in range(10):
    scores = net.classify_direct(lena_ready)
print 'Elapsed %s' % timer.total()

lena_ready = lena[np.newaxis, :227,:227].astype(np.float32)
print 'Testing direct classification with batches (10 runs)...'

for batch in [1,2,5,10,20,100]:
    lena_batch = np.tile(lena_ready, [batch, 1, 1, 1,])
    timer.reset()
    for i in range(10):
        scores = net.classify_direct(lena_batch)
    print 'Batch size %3d, equivalent time %s' % (batch, timer.total(False) / batch)

print 'Profiling (100 runs)...'
pr = profile.Profile()
pr.enable()
for i in range(100):
    scores = net.classify(lena)
pr.disable()
pr.dump_stats('lena_profile.pstats')
print 'Profiling done.'
