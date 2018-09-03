#!/bin/bash
# https://stackoverflow.com/questions/9998337/how-to-print-from-github
# can also use `pip install grip` or `npm install markdown-pdf`

pandoc -V geometry:margin=1in $1.md -f markdown --smart -s -o $1.pdf
