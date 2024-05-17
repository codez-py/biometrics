#import "@preview/polylux:0.3.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(footer: [I2B project])
// #enable-handout-mode(true)

#set text(size: 26pt)
#set strong(delta: 100)
#set par(justify: true)

#let frame(stroke) = (x, y) => (
  left: if x > 0 { 0pt } else { stroke }, right: stroke, top: if y < 2 { stroke } else { 0pt }, bottom: stroke,
)

#set table(fill: (rgb("EAF2F5"), none), stroke: frame(rgb("21222C")))
#show table.cell.where(y: 0): set text(weight: "bold")
#show table.cell: set text(size: 18pt)
#show figure.caption: set text(size: 18pt)
#show bibliography: set text(size: 20pt)

/******************************/

#title-slide(
  author: [
    Vimalraj S CS23M1007\
    E S Sarveswara Rao CS23M1008
  ], title: "Gender Classification using Gait", subtitle: "CS5014 - Introduction to Biometrics project", date: "15-05-2024",
)

#slide(title: "Overview")[
  #metropolis-outline
]

#new-section-slide("Introduction")
#slide(
  title: "Problem",
)[
  - Gender is an important fundamental attribute of humans.
  - Nowadays, most gender recognition methods are based on facial features.
  - This does not perform well when the subject is far away from cameras.
  - And when the resolution of the cameras is low.
  - We can use gait to recognize the gender.
]

#slide(
  title: "Applications",
)[
  - It can be used as a surveillance system for security.
  - It can also be used to classify customers in retail establishments.
]


#new-section-slide("Dataset")
#slide(
  title: "CASIA-B",
)[
  - Dataset B @shiqi2006 is a large multiview gait database.
  - Created in January 2005.
  - There are 124 subjects (93 males, 31 females).
  - Gait data was captured from 11 views.
  - Three variations, namely view angle, clothing and carrying condition changes,
    are separately considered.
  - Human silhouettes extracted from video files are also available.
]

#new-section-slide("Preprocessing")
#slide(
  title: "Gait Energy Image (GEI)",
)[
  - The silhouette is cropped and resized to make a fixed height of 224 pixels.
  - After that, the image is padded with black pixels to make its width 224 pixels.
  $ "GEI"(x, y) = sum_(t=1)^n I_t(x, y) $
]
#slide(title: "Multi channel GEI")[
  - Split the gait sequence into 3 parts.
  - Construct GEI for each sub sequence.
  - Merge the GEI's into a single image with 3 channels @kitchat2019.
]
#slide(title: "Examples")[
#grid(columns: 2,
  figure(image("out/gei.png"), caption: "Gait Energy Image"),
  figure(image("out-3/gei3.png"), caption: "3-channel GEI")
)]

#new-section-slide("Training")
#let angles = (90, 72, 54, 36, 18)

#slide(title: "Model")[
  #figure(image("out/model-rotated.png"), caption: "Model")
]

#slide(
  title: "Model",
)[
  A simple Convolutional Neural Network (CNN) with the following layers are used
  - Input layer of size 224 x 224
  - 3 blocks of Convolution, ReLU, MaxPooling
  - 2 Dense layers using ReLU activation
  - Output layer using Sigmoid activation
]

#slide(
  title: "Training",
)[
  - The model is trained separately for 5 different angles 90, 72, 54, 36, 18
  - Models are trained using both GEI and 3-channel GEI as input by changing only
    the input shape
  - The results of training are as follows
]

#for angle in angles {
  slide(
    title: [Training history for #angle#math.degree],
  )[

    #figure(
      image("out/nm-" + str(angle) + ".png"), caption: [Training history for #angle#math.degree, single channel GEI],
    )]

  slide(
    title: [Training history for #angle#math.degree],
  )[
    #figure(
      image("out-3/nm-" + str(angle) + ".png"), caption: [Training history for #angle#math.degree, 3 channel GEI],
    )
  ]
}

#new-section-slide("Evaluation")

#slide(
  title: [Evaluation Metrics],
)[
  #let comp = csv("out/comparison.trim.csv")
  #let comp3 = csv("out-3/comparison.trim.csv")

  #side-by-side[
    #figure(table(columns: 5, ..comp.flatten()), caption: "for single channel GEI")
    <fig:comparison>
  ][
    #figure(table(columns: 5, ..comp3.flatten()), caption: "for 3 channel GEI")
    <fig:comparison3>
  ]
]

#slide(
  title: "Evaluation Metrics",
)[
  - Accuracy, precision, recall and f1_score for different models are listed in
    @fig:comparison and @fig:comparison3
  - Accuracy and precision are important metrics for this task
  - Accuracy and precision decreases as the angle is shifted from 90#math.degree to
    18#math.degree
]

#new-section-slide("Conclusion & Future Scope")
#slide(
  title: "Conclusion",
)[
  - The model trained on 90#math.degree GEI's perform better than other models
  - Using 3-channel GEI's improve accuracy only at 90 degrees
  - 90#math.degree is the ideal angle to capture gait, but even for 72#math.degree gait
    sequence the model gives accuracy around 90%
]

#slide(
  title: "Future Scope",
)[
  - This approach can be extended to perform different types of classification
  - This simple model achieves good accuracy for the binary classification task.
  - Other complex models may be required to extract features in a multiclass
    classification problem.
]

#slide(title: "References")[
  #bibliography("refs.bib", title: none)
]

#focus-slide[
  The End
]
