#!/usr/bin/env bash

bxgrid -a hostname -u cvrl -p kodachrome \
biometrics export irises_still \
to bxgrid_data_full.csv \
/tmp/akuehlka as /sensorid/subjectid/sequenceid \
where TRUE AND \(sensorid in \(\'nd1N00020\',\'nd1N00049\',\'nd1N00074\',\'nd1N00077\',\'nd1N00079\'\)\)

#bxgrid -a hostname -u cvrl -p kodachrome \
#biometrics export_set livdet_2017_unknown_test \
#to livdet_2017_unknown_test.csv \
#/tmp/akuehlka as /sensorid/subjectid/sequenceid \
#where TRUE