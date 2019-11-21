import time


class Timer(object):
    def __init__(self, epoch, train_bn, val_bn, val_freq=1):
        self.epoch = epoch
        self.train_step = train_bn * epoch
        self.val_step = val_bn * epoch
        self.val_freq = val_freq
        self.val_eta_time = -1

    def eta(self, cur_step, step_time):
        if self.val_eta_time == -1:
            self.val_eta_time = step_time * self.val_step / self.val_freq
        eta = (self.train_step - cur_step) * step_time + self.val_eta_time
        return self.second2hour(eta)

    def set_val_eta(self, cur_epoch, val_time):
        self.val_eta_time = (self.epoch - cur_epoch) / self.val_freq * val_time

    def second2hour(self, s):
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return ("%02d:%02d:%02d" % (h, m, s))

if __name__ == "__main__":
    epoch = 5
    tbn = 200
    vbn = 70
    timer = Timer(epoch, tbn, vbn)
    for i in range(epoch):
        for j in range(tbn):

            # model every batch time
            start_time = time.time()
            for model in range(9999999):
                m = 28*99^32
            end_time = time.time()

            cur_step = j + i * tbn
            eta = timer.eta(cur_step, end_time - start_time)
            print(eta)
        # after every epoch
        start_time = time.time()
        for k in range(vbn):
            m = 99^2*32
        end_time = time.time()
        timer.set_val_eta(i, end_time - start_time)