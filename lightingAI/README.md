# LightningAI Guide

## Explanation of files:

There is 1 folder and 1 script in this directory.

``lightning_bench.py``: run this locally after you have setup the API endpoint on lightningAI


``lightningAI_Studio_Files``: These are files that you should upload or copy paste to the Studio on LightningAi, specific steps below:

## Step 1: 

Get a lightning AI account and get your free credits.

## Step 2:

Select start a new studio, and just choose AI Development.

## Step 3:

Copy paste or upload the files in ``lightningAI_Studio_Files`` to the studio, run server.py to make sure it works.

## Step 4:

Now return to the home page, and on the left go to ``deployment`` and select ``add deployment``

## Step 5:

Select your created studio and in the command section put in:

``` py
python server.py
```

and start the deployment with the GPU selection you'd like.

## Step 6:

Now run ``lightning_bench.py``.