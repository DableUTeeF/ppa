| weights                 | change             | val_acc |
|-------------------------|--------------------|---------|
|MobileNet_ppa_openimage_2| base for v3 data   | 0.8353/0.3812@11 0.6860 0.6310 0.1650 0.4167 0.0074|
|MobileNet_ppa_openimage_2| without augment    | 0.9061/0.4655@10 0.6938 0.6310 0.2388 0.4728 0.2863|
|MobileNet_ppa_openimage_3| use flipflop v1    | 0.8253/0.4023@11 0.7046 0.6707 0.1552 0.3142 0.1667|
|MobileNet_ppa_openimage_4| use shoeschanger   |         |
|MNet_ppa_oim_4           | use SGD            |         |
|MNet_ppa_oim_5           | use SGD + warmup   |         |


