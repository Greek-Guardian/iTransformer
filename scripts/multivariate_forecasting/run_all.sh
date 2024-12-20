mkdir -p log

# ECL
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/ECL/iTransformer.sh > log/iTransformer_ECL.log 2>&1

# ETT
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh > log/iTransformer_ETTh1.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTh2.sh > log/iTransformer_ETTh2.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTm1.sh > log/iTransformer_ETTm1.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTm2.sh > log/iTransformer_ETTm2.log 2>&1

# Exchange
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/Exchange/iTransformer.sh > log/iTransformer_Exchange.log 2>&1

# PEMS
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/PEMS/iTransformer_03.sh > log/iTransformer_03.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/PEMS/iTransformer_04.sh > log/iTransformer_04.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/PEMS/iTransformer_07.sh > log/iTransformer_07.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/PEMS/iTransformer_08.sh > log/iTransformer_08.log 2>&1

# SolarEnergy
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/SolarEnergy/iTransformer.sh > log/iTransformer_SolarEnergy.log 2>&1

# Traffic
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/Traffic/iTransformer.sh > log/iTransformer_Traffic.log 2>&1

# Weather
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/Weather/iTransformer.sh > log/iTransformer_Weather.log 2>&1