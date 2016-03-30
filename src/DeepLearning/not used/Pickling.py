import cPickle

#sace
save_file = open('path', 'wb')  # this will overwrite current contents
cPickle.dump(w.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
cPickle.dump(v.get_value(borrow=True), save_file, -1)  # .. and it triggers much more efficient
cPickle.dump(u.get_value(borrow=True), save_file, -1)  # .. storage than numpy's default
save_file.close()

#load
save_file = open('path')
w.set_value(cPickle.load(save_file), borrow=True)
v.set_value(cPickle.load(save_file), borrow=True)
u.set_value(cPickle.load(save_file), borrow=True)