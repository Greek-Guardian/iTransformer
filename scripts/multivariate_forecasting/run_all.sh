mkdir -p log_N_layers

# ECL
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/ECL/iTransformer.sh > log_N_layers/iTransformer_ECL.log 2>&1

# ETT
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh > log_N_layers/iTransformer_ETTh1.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTh2.sh > log_N_layers/iTransformer_ETTh2.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTm1.sh > log_N_layers/iTransformer_ETTm1.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTm2.sh > log_N_layers/iTransformer_ETTm2.log 2>&1

# Exchange
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/Exchange/iTransformer.sh > log_N_layers/iTransformer_Exchange.log 2>&1

# PEMS
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/PEMS/iTransformer_03.sh > log_N_layers/iTransformer_03.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/PEMS/iTransformer_04.sh > log_N_layers/iTransformer_04.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/PEMS/iTransformer_07.sh > log_N_layers/iTransformer_07.log 2>&1
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/PEMS/iTransformer_08.sh > log_N_layers/iTransformer_08.log 2>&1

# SolarEnergy
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/SolarEnergy/iTransformer.sh > log_N_layers/iTransformer_SolarEnergy.log 2>&1

# Traffic
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/Traffic/iTransformer.sh > log_N_layers/iTransformer_Traffic.log 2>&1

# Weather
bash /root/workspace/iTransformer/scripts/multivariate_forecasting/Weather/iTransformer.sh > log_N_layers/iTransformer_Weather.log 2>&1