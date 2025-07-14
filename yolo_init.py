# --- YOLOv5 TFLite Initialization ---
        self.interpreter = tf.lite.Interpreter(model_path="yolov5s_f16.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1]
        self.int8 = self.input_details[0]["dtype"] == np.uint8

        self.label_names = [
            "person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign"
        ]

