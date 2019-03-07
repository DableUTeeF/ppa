| weights                 | change             |  val_acc  | map    | person | helmet | goodshoes | badshoes | LP   |
|-------------------------|--------------------|-----------|--------|--------|--------|-----------|----------|------|
|MobileNet_ppa_openimage_2| base for v3 data   | 0.8353@11 | 0.3812 | 0.6860 | 0.6310 | 0.1650    | 0.4167   |0.0074|
|MobileNet_ppa_openimage_2| without augment**  | 0.9061@10 | 0.4655 | 0.6938 | 0.6310 | 0.2388    | 0.4728   |0.2863|
|MobileNet_ppa_openimage_3| use flipflop v1    | 0.8253@11 | 0.4023 | 0.7046 | 0.6707 | 0.1552    | 0.3142   |0.1667|
|MobileNet_ppa_openimage_4| use shoeschanger   |           |
|MNet_ppa_oim_4           | use SGD            | 0.8771@9  | 0.3776 | 0.6411 | 0.6534 | 0.1572    | 0.3482   |0.0882|
|MNet_ppa_oim_5           | Continue with Adam | 0.8776@2  | 0.4186 | 0.6583 | 0.6909 | 0.2093    | 0.3933   |0.1413|
|MNet_ppa_oim_6           | Use zero padding   | 0.6140@15 | 0.3954 | 0.7205 | 0.6040 | 0.1897    | 0.3610   |0.1017|
|MNet_ppa_oim_7           | reduce box per im  | 0.9138@9  | 0.3910 | 0.6415 | 0.6728 | 0.2463    | 0.3454   |0.0490|
|MNet_ppa_oim_8           | increase box       | 0.8792@14 | 0.4393 | 0.7394 | 0.6755 | 0.2058    | 0.3701   |0.2059|
|MNet_ppa_oim_9           | use yolo v3        |           |
|MNet_ppa_oim_10          | yolov3 new scale   |           |



