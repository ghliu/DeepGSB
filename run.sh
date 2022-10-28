
DATE=$(date +%m.%d)

EXP=$1

if [[ "$EXP" == "GMM" || "$EXP" == "all" ]]; then
    # GMM (std1, obs1500, multistep, gauss)
    BASE='--problem-name GMM --ckpt-freq 2 --snapshot-freq 2 --log-tb'
    python main.py $BASE --sb-param critic       --dir gmm-$DATE/deepgsb-c-std1
    python main.py $BASE --sb-param actor-critic --dir gmm-$DATE/deepgsb-ac-std1
fi

if [[ "$EXP" == "Vneck" || "$EXP" == "all" ]]; then
    # Vneck (std1, obs1500, mf0/3, multistep, jacobi, use_rb)
    BASE='--problem-name Vneck --ckpt-freq 2 --snapshot-freq 2 --log-tb'
    python main.py $BASE --sb-param critic       --dir vneck-$DATE/deepgsb-c-std1-mf3  --MF-cost 3.0
    python main.py $BASE --sb-param critic       --dir vneck-$DATE/deepgsb-c-std1-mf0  --MF-cost 0.0

    python main.py $BASE --sb-param actor-critic --dir vneck-$DATE/deepgsb-ac-std1-mf3 --MF-cost 3.0
    python main.py $BASE --sb-param actor-critic --dir vneck-$DATE/deepgsb-ac-std1-mf0 --MF-cost 0.0
fi

if [[ "$EXP" == "Stunnel" || "$EXP" == "all" ]]; then
    # Stunnel (obs1500, congestion1, multistep, jacobi, use_rb)
    BASE='--problem-name Stunnel --ckpt-freq 2 --snapshot-freq 2 --log-tb --MF-cost 0.5 '

    python main.py $BASE --sb-param critic       --dir stunnel-$DATE/deepgsb-c-std0.5-mf1 --diffusion-std 0.5
    python main.py $BASE --sb-param critic       --dir stunnel-$DATE/deepgsb-c-std1-mf1   --diffusion-std 1.0
    python main.py $BASE --sb-param critic       --dir stunnel-$DATE/deepgsb-c-std2-mf1   --diffusion-std 2.0

    python main.py $BASE --sb-param actor-critic --dir stunnel-$DATE/deepgsb-ac-std0.5-mf1 --diffusion-std 0.5
    python main.py $BASE --sb-param actor-critic --dir stunnel-$DATE/deepgsb-ac-std1-mf1   --diffusion-std 1.0
    python main.py $BASE --sb-param actor-critic --dir stunnel-$DATE/deepgsb-ac-std2-mf1   --diffusion-std 2.0
fi

if [[ "$EXP" == "opinion" || "$EXP" == "all" ]]; then
    # opinion (std0.1, multistep, gauss, use_rb)
    BASE='--problem-name opinion --ckpt-freq 10 --snapshot-freq 2 --log-tb --MF-cost 1.0 --weighted-loss'
    python main.py $BASE --sb-param critic       --dir opinion-2d-$DATE/deepgsb-c-std0.1-mf1-w
    python main.py $BASE --sb-param actor-critic --dir opinion-2d-$DATE/deepgsb-ac-std0.1-mf1-w
fi

if [[ "$EXP" == "opinion-1k" || "$EXP" == "all" ]]; then
    # opinion (std0.5, singlestep, gauss)
    BASE='--problem-name opinion_1k --ckpt-freq 20 --snapshot-freq 10 --log-tb --weighted-loss'
    python main.py $BASE --sb-param actor-critic --dir opinion-1k-$DATE/deepgsb-ac-std0.5-mf1-w --MF-cost 1.0
    python main.py $BASE --sb-param actor-critic --dir opinion-1k-$DATE/deepgsb-ac-std0.5-mf0-w --MF-cost 0.0
fi
