import cv2
import torch
import os
import numpy as np
import argparse


from segmentation_network.segnet import Network
from db import Database
import ImageProcessing as ip


def main(paintings_dir, csv_file, videos_dir, frame_interval, resize):

    frame = None
    ret = False
    index = 0
    interval = frame_interval
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    # Defining the computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Loading the segmentation network
    net = Network("config/ade20k-hrnetv2.yaml", device)
    # Reading the csv file
    database = Database(csv_file, paintings_dir)
    # Computing the keypoints of the images in the db
    database.computeKeypoints(2)

    # Iterating over all the videos in the data directory
    for file_name in os.listdir(videos_dir):

        # Opening the video
        video = cv2.VideoCapture(videos_dir + "/" + file_name)

        if video.isOpened():
            ret, frame = video.read()
        else:
            print("Unable to load the video: ", video.get("name"))
            exit(1)

        room = "Room not identificated"

        while ret:

            if index == 0:

                # Resizing
                frame = ip.resize(frame, resize)
                # Building data loader for the network
                loader = net.build_loader([frame[:, :, ::-1]])

                # Passing the frame through the network
                results = net.evaluate(loader)
                # Visualizing the results
                net.visualize_result("frame", frame, results[0])
                # Retrieving contours of paintings
                contours = get_object_contours(net, results, "painting", 4000)
                # Approximation of the contours
                contours = ip.rectify_contours(contours)

                # rectifying the paintings
                copy_frame = frame.copy()
                projections = []
                for i, contour in enumerate(contours):
                    if contour.shape[0] == 4:
                        projections.append(ip.project_image(copy_frame, contour[:, 0, :], "Projected " + str(i)))
                    else:
                        projections.append("No squared")

                cv2.drawContours(copy_frame, contours, -1, (0, 255, 0), 3)
                # finding the correspondecies in the database
                paintings = []
                title = "Not found"
                for proj_index, proj in enumerate(projections):
                    if proj != "No squared":
                        proj = ip.linear_filter(proj)
                        kp, des = orb.detectAndCompute(proj, None)
                        matches_found = np.zeros(database.length())
                        for i in range(database.length()):
                            painting_kp = database.getKeypoints(i)
                            matches = matcher.match(des, painting_kp[1])
                            sorted(matches, key=lambda element: element.distance)
                            good_matches = []
                            for j, m in enumerate(matches):
                                if j < len(matches) - 1 and m.distance < 0.7 * matches[j + 1].distance:
                                    good_matches.append(m)
                            matches_found[i] = len(good_matches)


                        sorted_matches = sorted(matches_found, reverse=True)[:5]
                        if np.max(matches_found) > 8 and sorted_matches[0] - sorted_matches[1] > 5:
                            index = np.argmax(matches_found)
                            title = database.getTitle(index) + "-" + database.getImageName(index)
                            paintings.append(index)

                    coord = np.argmin(np.sum(contours[proj_index], axis=(1, 2)))
                    cv2.putText(copy_frame, title, (contours[proj_index][coord][0][0] + 10, contours[proj_index][coord][0][1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

                if len(paintings) > 0:
                    print("Paintings found: ", len(paintings))
                    for painting in paintings:
                        print(painting, ": ", database.getTitle(painting), " - ", database.getAuthor(painting))
                        room = "Room " + database.getRoom(painting)
                        #cv2.imshow(database.getTitle(painting), database.getImage(painting))
                else:
                    print("No painting recognized")

                # Retrieving the contours of the persons
                contours = get_object_contours(net, results, "person", 400)
                #compute bounding boxes
                boxes = ip.compute_bounding_boxes(contours)
                # Drawing the bounding boxes
                result_img = ip.draw_bounding_boxes(np.copy(copy_frame), boxes, (255, 200, 0), 2)
                for box in boxes:
                    x, y, w, h = box
                    cv2.putText(result_img, room, (x + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0))
                cv2.imshow("boxes", result_img)

            # Setting q as exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Reading next frame
            ret, frame = video.read()
            index = (index + 1) % interval

        video.release()
        cv2.destroyAllWindows()


def get_object_contours(net, mask, obj, min_size):
    # Extracting the objects requested
    objects = net.extract_object(mask, obj)
    # Finding the connected components
    components_with_stats = cv2.connectedComponentsWithStats(objects, connectivity=8)
    # Removing the small segmented objects from the image
    cleaned_image = ip.remove_small_components(components_with_stats, min_size=min_size)
    # Finding contours
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="path to paintings_db directory")
    ap.add_argument("-f", "--csv_file", required=True, help="path to .csv file")
    ap.add_argument("-v", "--videos", required=True, help="path of the videos to submit")
    ap.add_argument("-fr", "--frame_interval", type=int, default=30, help="number of frame skipped")
    ap.add_argument("-r", "--resize", type=int, default=2, help="dimension of video resize")
    args = vars(ap.parse_args())

    main(args['images'], args['csv_file'], args['videos'], args['frame_interval'], args['resize'])
