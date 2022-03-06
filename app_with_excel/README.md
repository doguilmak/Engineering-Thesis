
#  Developing Application that Calculates Legendre Differential Equations and Can Save to Excel Table with Python 

## Statement

The aim of the project, as can be understood from the title, is to developing an application that can calculate ***legendre differential equations with Python 3.7*** and save them in Excel Table. These operations were carried out through the ***PyQt5*** library and the ***Designer*** application.

## Methodology

### Unnormalized function

![Unnormalized function](unnormalized.jpg)

### Normalized function

![Normalized function](normalized.png)

In mathematics, the associated legendre polynomials are the canonical solutions of the general Legendre equation.

This equation has nonzero solutions that are nonsingular on (−1, 1) only if n and m are integers with 0 ≤ m ≤ n, or with trivially equivalent negative values. When in addition m is even, the function is a polynomial. When m is zero and n integer, these functions are identical to the Legendre polynomials.

In this part of work we are going to calculate the values of the Pnm normalized values. ***normalized.png*** is the mathematical equation of normalized legendre equation and ***unnormalized.jpg***  is the mathematical equation of unnormalized legendre equation. **Normalized values  were calculated in project.**

**UI Overview:**

![UI](UI.jpg)

## Analysis

You can examine the excel output, where **n and m values ​​are 10**, on the **output.xlsx** file. The same output is also shown in the application interface. In order for it to be saved as Excel as optional, it must first be written in the relevant field, indicating the name of the file. (for example: '***C:\Users\User\Desktop\output.xlsx***')

## How to Run Code

Before running the code make sure that you have these libraries:

 - PyQt5
 - numpy
 - math
 - pandas

## File Conversions

Before doing these operations, **you need to go to the location on command prompt where the Calculating_Pnm_excel.ui and Calculating_Pnm_excel.py files are located.** These procedures were done by myself. You are free to make .exe file from .py file. Those who are curious about how it is made or who want to learn can try this section on their own.

### .ui to .py

    python -m PyQt5.uic.pyuic -x Calculating_Pnm_excel.ui -o Calculating_Pnm_excel.py

### .py to .exe

    pyinstaller.exe --onefile --windowed --icon=app.ico Calculating_Pnm_excel.py

## Contact Me

If you have something to say to me please contact me: 

 - Twitter: [Doguilmak](https://twitter.com/Doguilmak)  
 - Mail address: doguilmak@gmail.com
 
