#!/usr/bin/python
import pyfiglet
from db_models.mongo_setup import global_init
from db_models.models.cache_model import Cache
import uuid
import globals
import init
from image_recog_service import predict
import requests
from init import err_logger


global_init()

FILE_ID = ""


def save_to_db(db_object, labels, scores):
    try:
        print("*****************SAVING TO DB******************************")
        print("in save")
        print(db_object)
        print(db_object.id)
        db_object.labels = labels
        db_object.scores = scores
        db_object.save()
        print("*****************SAVED TO DB******************************")
    except Exception as e:
        print(" ERROR IN SAVE TO DB")
        err_logger(str(e)+" ERROR IN SAVE TO DB")


def update_state(file_name):
    payload = {
        'parent_name': globals.PARENT_NAME,
        'group_name': globals.GROUP_NAME,
        'container_name': globals.RECEIVE_TOPIC,
        'file_name': file_name,
        'client_id': globals.CLIENT_ID
    }
    try:
        requests.request("POST", globals.DASHBOARD_URL,  data=payload)
    except Exception as e:
        print(f"{e} EXCEPTION IN UPDATE STATE API CALL......")
        err_logger(f"{e} EXCEPTION IN UPDATE STATE API CALL......FILE ID {FILE_ID}")


if __name__ == '__main__':
    print(pyfiglet.figlet_format(str(globals.RECEIVE_TOPIC)))
    print(pyfiglet.figlet_format("INDEXING CONTAINER"))
    print("Connected to Kafka at " + globals.KAFKA_HOSTNAME + ":" + globals.KAFKA_PORT)
    print("Kafka Consumer topic for this Container is " + globals.RECEIVE_TOPIC)
    for message in init.consumer_obj:
        message = message.value
        db_key = str(message)
        print(db_key, 'db_key')
        FILE_ID = db_key
        try:
            db_object = Cache.objects.get(pk=db_key)
            file_name = db_object.file_name
            print("#############################################")
            print("########## PROCESSING FILE " + file_name)
            print("#############################################")
            final_labels = db_object.labels
            final_scores = db_object.scores
        except Exception as e:
            print("EXCEPTION IN FETCHING FROM DATABASE......")
            err_logger(str(e) + " EXCEPTION IN FETCHING FROM DATABASE......FILE ID " + FILE_ID)
            continue

        if db_object.is_doc_type:
            """document"""
            images_array = []
            for image in db_object.files:
                pdf_image = str(uuid.uuid4()) + ".jpg"
                with open(pdf_image, 'wb') as file_to_save:
                    file_to_save.write(image.file.read())
                images_array.append(pdf_image)
            for image in images_array:
                try:
                    response = predict(file_name=image)
                    for label, score in zip(response["labels"], response['scores']):
                        if label not in final_labels:
                            final_labels.append(label.strip())
                            final_scores.append(score)
                        else:
                            x = final_labels.index(label)
                            score_to_check = final_scores[x]
                            if score > score_to_check:
                                final_scores[x] = score
                    save_to_db(db_object, final_labels, final_scores)
                    print(".....................FINISHED PROCESSING FILE.....................")
                    update_state(file_name)
                except Exception as e:
                    print(str(e) + "Exception in predict")
                    err_logger(str(e) + "Exception in predict")
                    continue
        else:
            """image"""
            with open(file_name, 'wb') as file_to_save:
                file_to_save.write(db_object.file.read())
            try:
                image_results = predict(file_name)
                for label, score in zip(image_results["labels"], image_results['scores']):
                    if label not in final_labels:
                        final_labels.append(label.strip())
                        final_scores.append(score)
                    else:
                        x = final_labels.index(label)
                        score_to_check = final_scores[x]
                        if score > score_to_check:
                            final_scores[x] = score

                save_to_db(db_object, final_labels, final_scores)
                print(".....................FINISHED PROCESSING FILE.....................")
                update_state(file_name)
            except Exception as e:
                print(str(e) + " Exception in predict")
                err_logger(str(e) + " Exception in predict")
