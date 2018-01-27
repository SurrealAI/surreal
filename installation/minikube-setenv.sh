#!/bin/bash
HELPER=minikube-setenv-helper.sh

if [ ! -z "$1" ]; then
    echo "Writing setenv to ~/.$1"
    cat $HELPER >> ~/.$1
else
    chmod +x $HELPER
    ./$HELPER
fi

