### 1. `train/test split` 비율 변경

**`test_size=0.5` → `test_size=0.2`**

훈련 데이터의 양이 2배 많아졌기 때문에, 모델이 더 다양한 패턴을 학습

---

### 2. `num_virtual_tokens=20` → `num_virtual_tokens=40`

프롬프트의 표현력이 증가하면서, 문제에 더 정밀하게 대응

---

### 3. `num_train_epochs=1` → `num_train_epochs=5`

모델이 훈련 데이터에 대해 충분한 학습
