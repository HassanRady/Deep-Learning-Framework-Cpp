
class BaseLayer {
protected: BaseLayer() {
        trainable = false;
        initializable = false;
    }

    bool trainable;
    bool initializable;
};