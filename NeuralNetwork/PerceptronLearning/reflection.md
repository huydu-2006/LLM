        w ban đầu
            ↓
        Dự đoán X
            ↓
        Có điểm nào sai?
       ↙                ↘
     Có                 Không
      ↓                   ↓
    Chọn 1 điểm sai    return w
      ↓
    Cập nhật w
      ↓
    Quay lại dự đoán

Perceptron cho mình hiểu nền tảng:

$$ \boxed{x \rightarrow w^Tx \rightarrow activation \rightarrow prediction \rightarrow error \rightarrow update\ w} $$

Sau này khi học neural network, ý tưởng forward → tính loss/error → backpropagation → update weights là sự phát triển phức tạp hơn rất nhiều của tư tưởng này.