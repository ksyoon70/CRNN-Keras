#아래 label 의 정의는 prediction.py에 나옴.
#다만 여기서는 이 레이블링을 쓰지 않고 새로 정의하여 씀.
#CHAR_VECTOR = "adefghjknqrstwABCDEFGHIJKLMNOPZ0123456789"
#한국번호판에서 지역문자 제외 하고 나올수 있는 문자임.
CHAR_VECTOR = "0123456789가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히육해공국합배"

letters = [letter for letter in CHAR_VECTOR]

num_classes = len(letters) + 1

#img_w, img_h = 128, 64
#가지고 있는 영상이 SSD 320x320 이므로 수정함.
img_w, img_h = 224, 224

# Network parameters
batch_size = 16
val_batch_size = 16

downsample_factor = 4
#max_text_len = 9
max_text_len = 1 #한줄번호판의 최대길이는 7글자 이므로 약간 수정함.