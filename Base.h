#pragma once

class BaseLayer {
protected:
    BaseLayer() {
        trainable = false;
        initializable = false;
        training = true;
    }

    int train() {
        training = true;
    }

    int eval() {
        training = false;
    }

    bool trainable;
    bool initializable;
    bool training;
};