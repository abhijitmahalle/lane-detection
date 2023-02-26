[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Lane Detection and Turn Prediction  
This repository contains code to detect lanes on straight and curved roads using classical approach of computer vision to mimic Lane Departure Warning System in self-driving cars. Concepts of homography, polynomial curve fitting, hough lines, warping, and unwarping have been implemented to get the results. The program also predicts turn on curved roads by computing the radius of curvature.

## Pipeline

### Straight Lane Detection
**1. Solid Lane Detection:**
  - Apply mask to detect region of interest(part of the image containing lane to be detected).
  - Apply [Hough Transform](https://en.wikipedia.org/wiki/Hough_transform) with minimum length value(decided by tuning) high enough to detect only solid lane.
  - Find the mean of all the start and end points.
  - Find the slope mean value calculated in step 2.
  - Draw a single line on the image with calculated slope.
  
 **2. Dashed Lane Detection:**
  - Apply mask to detect region of interest(part of the image containing lane to be detected).
  - Apply Hough transform to detect all the line in the masked image.
  - Find the slope of all the lines and remove those lines which is positive slope if solid lane has positive slope and negative slope if solid line has negative slope.
  - Find the mean of all the start and end points in the remaining lines.
  - Find the slope with mean value calculated in step 3.
  - Draw a single line with calculated slope.
  
 ### Curve Lane Detection and Turn Prediction
 **1. White Lane Detection:**
  -  Compute the Homography and perform warp perspective to take bird eye view of the region of interest (part of the image containing lane to be detected).
  - Thresholding to remove yellow lane and noise from the warped image. The output will be binary image containing only white lane.
  - Find the pixel coordinates having value 255.
  - Find the equation of curve from above pixel coordinate.
  - Extrapolate and plot the white lane using above equation.
  - Compute the radius of curvature using [equation of curve](https://www.cuemath.com/radius-of-curvature-formula/).
  
 **2. Yellow Lane Detection:**
  - convert the image into hsv color space and apply color mask to detect only yellow lane.
  - Find the pixel coordinate of the yellow pixel.
  - Find the equation of curve from above pixel coordinate.
  - Extrapolate and plot the yellow lane using above equation.
  - Compute the radius of curvature using equation of curve.
  
 **3. Radius of Curvature:**
  - Take the average of white and yellow lane radius.
  
 **4. Direction of Turn:**
  - If the coefficient of highest degree term in the equation of white/yellow lane is positive, then the turn would be right. If it is negative, then the turn would be left and if it is zero then then is a no turn.

## Requirement:
- Python 2.0 or above

## Dependencies:
- OpenCV
- NumPy

## Instructions to run the code:
Run the following command in the terminal:
```
python straight_lane_detection.py  
python curved_lane_detection.py
```

## Result
![](https://github.com/abhijitmahalle/lane_detection/blob/master/gif/curved_lane_detection.gif)

## How well the solution can generalize?
### Straight Lane Detection
The solution will work in the image that satisfies the following scenario:
 - One line should be solid, and another should be dashed irrespective of the left or right position.
 - The shade of the color of the road should not be significantly different than the given shade.
 - Field of view and direction of view of the video should not be significantly different from the given video.

### Curve Lane Detection and Turn Prediction
The solution will work in the image that satisfy following scenario:
 - One line should be yellow, and another should be white irrespective of the left or right position or whether they are solid or dashed or whether they are curved or straight.
 - Field of view and direction of view of the video should not be significantly different than given video.
