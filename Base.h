//
// Created by hassan on 22.04.23.
//

#ifndef CNN_IN_C___BASE_H
#define CNN_IN_C___BASE_H

#endif //CNN_IN_C___BASE_H

class BaseLayer {
protected: BaseLayer() {
        trainable = false;
        initializable = false;
    }

    bool trainable;
    bool initializable;
};