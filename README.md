# https起動時コマンド:
    python manage.py runsslserver 0.0.0.0:8000 --certificate ../foobar.crt --key ../foobar.key

# iPhone側での接続方法:
1. macのterminalでipconfig getifaddr en0を実行

2.https://(取得したipアドレス):8000でつなぐ
