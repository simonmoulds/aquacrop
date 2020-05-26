#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import test_utils

def test_AgMERRA_Ghana_Maize(tmpdir, context):
    aqpy_yld, aos_yld = test_utils.read_multiyear_yield(tmpdir)
    np.testing.assert_almost_equal(aqpy_yld, aos_yld, 1)
