#!/usr/bin/python
import json
from db_models.mongo_setup import global_init
from db_models.models.cache_model import Cache
import uuid
import globals
import init
from image_recog_service import predict


def send_to_topic(topic, value_to_send_dic):
    data_json = json.dumps(value_to_send_dic)
    init.producer_obj.send(topic, value=data_json)



if __name__ == '__main__':
    for message in init.consumer_obj:
        global_init()
        message = message.value
        db_key = str(message)
        db_object = Cache.objects.get(pk=db_key)
        file_name = db_object.file_name
        init.redis_obj.set(globals.RECEIVE_TOPIC, file_name)
        if db_object.is_doc_type:
            """document"""
            images_array = []
            for image in db_object.files:
                pdf_image = str(uuid.uuid4()) + ".jpg"
                with open(pdf_image, 'wb') as file_to_save:
                    file_to_save.write(image.file.read())
                images_array.append(pdf_image)
            full_results = []
            text_results = []
            for image in images_array:
                response = predict(file_name=image, doc=True)
                full_results_dict = response["full_results"]
                text_results_dict = response["text"]
                full_results.append(full_results_dict)
                text_results.append(text_results_dict)
            final_full_response = {
                "container_name": globals.RECEIVE_TOPIC,
                "file_name": file_name,
                "results": full_results,
                "is_doc_type": True
            }
            final_text_response = {
                "container_name": globals.RECEIVE_TOPIC,
                "file_name": file_name,
                "results": text_results,
                "is_doc_type": True
            }
            send_to_topic(globals.SEND_TOPIC_FULL, value_to_send_dic=final_full_response)
            send_to_topic(globals.SEND_TOPIC_TEXT, value_to_send_dic=final_text_response)
            init.producer_obj.flush()
        else:
            """image"""
            if db_object.mime_type in globals.ALLOWED_IMAGE_TYPES:
                with open(file_name, 'wb') as file_to_save:
                    file_to_save.write(db_object.file.read())
                image_results = predict(file_name)
                full_results_dict = image_results["full_results"]
                text_results_dict = image_results["text"]
                full_results_dict["container_name"] = globals.RECEIVE_TOPIC
                text_results_dict["container_name"] = globals.RECEIVE_TOPIC
                send_to_topic(globals.SEND_TOPIC_FULL, value_to_send_dic=full_results_dict)
                send_to_topic(globals.SEND_TOPIC_TEXT, value_to_send_dic=text_results_dict)
                init.producer_obj.flush()

