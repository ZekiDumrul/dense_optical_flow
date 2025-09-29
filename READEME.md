# Optical Flow ile El Hareketi Takibi

Bu proje, **webcam kullanarak gerçek zamanlı el hareketi takibi** yapar.  
Takip için iki farklı yöntem kullanılabilir ve kullanıcı bu yöntemleri çalışırken seçebilir:

- **Lucas-Kanade Optical Flow** (özellik tabanlı)
- **Farneback Optical Flow** (yoğun optik akış)

Ayrıca sistem, **Haar Cascade sınıflandırıcısı** ile yüzü tespit edip hariç tutar. Böylece sadece **el hareketleri** takip edilir.

---

## Özellikler
- Kameradan canlı görüntü alır.
- Yöntem seçimini klavye ile yapabilirsiniz:
  - **`l` tuşu** → Lucas-Kanade yöntemi
  - **`f` tuşu** → Farneback yöntemi
- ROI (**Region of Interest**) seçerek takip edilecek el bölgesini belirleyebilirsiniz.
- Yüz bölgeleri otomatik olarak hariç tutulur.
- Hareketli alanlar dikdörtgen kutu ve “El” etiketiyle gösterilir.
- **`ESC` tuşu** ile çıkış yapılır.

---

## Gereksinimler
- Python 3.x
- [OpenCV](https://opencv.org/) (`cv2`)
- `numpy`

Gerekli kütüphaneleri yüklemek için:
```bash
pip install opencv-python numpy
