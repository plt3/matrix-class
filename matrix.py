from copy import deepcopy


# Mat class creates matrix objects from a list of lists, defines:
# matrix addition, subtraction, multiplication, scalar multiplication,
# transpose, row-echelon form (REF), reduced row-echelon form (RREF),
# inverse, determinant, and lets user input matrix values quickly using
# Mat.from_input() class method

class Mat:
    def __init__(self, valuelist):
        for row in valuelist:
            for entry in row:  # raise error if user inputs non-numeric values
                if not isinstance(entry, (int, float)):
                    raise ValueError('Matrix can only contain numbers.')

            if len(row) != len(valuelist[0]):  # raise error if matrix isn't rectangular
                raise IndexError('All rows must have the same number of entries.')

        Mat.roundnums(valuelist)  # round entries

        self.valuelist = valuelist

    def __str__(self):  # visual representation of Mat objet
        roundedvals = [[round(i, 4) for i in row] for row in self.valuelist]

        if len(roundedvals) == 1:  # case if matrix only has one row
            formatted = ' '.join([str(a) for a in roundedvals[0]])

            return f'[{formatted}]'

        mostchars = 0

        for row in roundedvals:
            for num in row:
                if len(str(num)) > mostchars:
                    mostchars = len(str(num))  # for centering purposes

        visual = ''

        for number, row in enumerate(roundedvals):
            if number == 0:
                formatrow = '⌈'  # create brackets around matrix
            elif number == (len(roundedvals) - 1):
                formatrow = '⌊'
            else:
                formatrow = '|'

            for value in row:  # center values
                formatrow += str(value).center(mostchars + 2)

            if number == 0:
                visual += formatrow + '⌉\n'
            elif number == (len(roundedvals) - 1):
                visual += formatrow + '⌋\n'
            else:
                visual += formatrow + '|\n'

        return visual.rstrip()

    def __add__(self, other):  # define matrix addition
        if not self.samedimensions(other):  # raise error if matrice aren't same size
            raise TypeError('The sum of these matrices is undefined.')

        endlist = []

        for row, otherrow in zip(self.valuelist, other.valuelist):
            newrow = []

            for value, othervalue in zip(row, otherrow):
                newrow.append(value + othervalue)  # add each value

            endlist.append(newrow)

        object = Mat(endlist)

        return object

    def __sub__(self, other):  # define matrix subtraction
        if not self.samedimensions(other):  # raise error if matrice aren't same size
            raise TypeError('The difference of these matrices is undefined.')

        endlist = []

        for row, otherrow in zip(self.valuelist, other.valuelist):
            newrow = []

            for value, othervalue in zip(row, otherrow):
                newrow.append(value - othervalue)  # subtract each value

            endlist.append(newrow)

        object = Mat(endlist)

        return object

    def __mul__(self, other):  # define matrix multiplication
        if isinstance(other, int):  # define scalar multiplication (only Mat * scalar)
            return Mat([[other * ent for ent in row] for row in self.valuelist])

        newother = other.transpose()

        for row, otherrow in zip(self.valuelist, newother.valuelist):
            if len(row) != len(otherrow):  # raise error if product is undefined
                raise TypeError('The product of these matrices is undefined.')

        endlist = []

        for row in self.valuelist:
            endrow = []
            for otherrow in newother.valuelist:
                total = 0

                for value, othervalue in zip(row, otherrow):
                    total += value * othervalue  # row-by-column dot product

                endrow.append(total)

            endlist.append(endrow)

        object = Mat(endlist)

        return object

    @staticmethod
    def roundnums(rlist):  # round so that -0.0 is recognized as 0
        for row in rlist:
            for index, entry in enumerate(row):
                if abs(entry - round(entry)) < 0.000000001:  # arbitrary error margin
                    row[index] = round(entry)

        return rlist

    @staticmethod
    def raw_transpose(rowlist):  # transpose a list of lists
        newvaluelist = [[a[i] for a in rowlist] for i in range(len(rowlist[0]))]

        return newvaluelist

    def transpose(self):  # create new Mat object that is transpose of self
        columnslist = Mat.raw_transpose(self.valuelist)

        return Mat(columnslist)

    @staticmethod  # swap rows and create zeros below pivot position (first step of row reduction, called by Mat.raw_ref(list))
    def makezerosbelow(rowlist, swaps=False):  # optional argument is only used by self.det() through Mat.raw_ref(list)
        rows = rowlist
        columns = Mat.raw_transpose(rows)

        swapcount = 0  # for use in calculating determinant

        for colind, column in enumerate(columns):  # swap rows to get nonzero at top of column
            for index, entry in enumerate(column):
                if entry != 0:
                    pivotcol = colind
                    pivotval = entry
                    rows[0], rows[index] = rows[index], rows[0]  # swap rows (row interchange)

                    if index != 0:
                        swapcount += 1  # add swap to get correct sign of determinant

                    break
            else:
                continue
            break  # break out of both loops

        for row in rows[1:]:  # create zeros below the pivot position (row replacement)
            try:
                coef = -1 * row[pivotcol] / pivotval
            except UnboundLocalError:  # in case of zero matrix
                coef = 0

            for index, entry in enumerate(row):
                row[index] += coef * rows[0][index]  # row replacement

        Mat.roundnums(rows)

        if swaps:  # only for calculating determinants
            return (swapcount, rows)
        else:
            return rows

    @staticmethod
    def raw_ref(rowlist, swaps=False):  # returns row-echelon form of list of lists, called by self.ref()
        submatrix = deepcopy(rowlist)  # to avoid altering rowlist
        refvalues = []

        totalswaps = 0  # only used to calculate determinant

        while len(submatrix) >= 1:
            if swaps:
                zeros = Mat.makezerosbelow(submatrix, True)  # for determinant
                refvalues.append(zeros[1][0])
                submatrix = zeros[1][1:]
                totalswaps += zeros[0]
            else:
                zeros = Mat.makezerosbelow(submatrix)
                refvalues.append(zeros[0])
                submatrix = zeros[1:]  # cut off top row until only one row left

        if swaps:
            return (totalswaps, refvalues)  # for determinant use
        else:
            return refvalues

    def ref(self):  # return Mat object of self in row-echelon form
        refmat = Mat.raw_ref(self.valuelist)

        return Mat(refmat)

    @staticmethod
    def makezerosabove(rowlist):  # for use in Mat.raw_rref(list), create zeros above pivot position
        echelon = rowlist

        pivotrow = 0

        for rowindex, row in enumerate(echelon[::-1]):
            for index, entry in enumerate(row):
                if entry != 0:
                    pivotcol = index  # save location of pivot position
                    pivotrow = len(echelon) - rowindex - 1

                    if entry != 1:  # make leading entry in row become 1
                        for secondindex, number in enumerate(row[index:]):
                            row[index + secondindex] = number / entry
                    break
            else:
                continue
            break  # break out of both loops

        for row in echelon[:pivotrow]:  # make zeros above pivot position
            try:
                coef = -1 * row[pivotcol]
            except UnboundLocalError:
                coef = 0  # in case of zero matrix
            for index, entry in enumerate(row[pivotcol:]):
                row[index + pivotcol] += coef * echelon[pivotrow][index + pivotcol]

        Mat.roundnums(echelon)

        return echelon

    @staticmethod
    def raw_rref(rowlist):  # called by self.rref(), creats RREF list from inputted list
        start = Mat.raw_ref(rowlist)  # start by getting to REF
        rrefvalues = []

        while len(start) >= 1:
            zeros = Mat.makezerosabove(start)

            rownum = 0

            for index, row in enumerate(zeros[::-1]):  # iterate backwards to start at last row
                for value in row:
                    if value != 0:
                        rrefvalues.append(row)
                        rownum = len(zeros) - index - 1  # get pivot row number
                        break
                else:
                    rrefvalues.append(row)  # in case of zero matrix
                    continue
                break

            start = zeros[:rownum]

        return rrefvalues[::-1]  # reverse list to get final RREF

    def rref(self):  # returns Mat object of self in RREF
        endvalues = Mat.raw_rref(self.valuelist)

        return Mat(endvalues)

    def det(self):  # calculate determinant of matrix
        swaps, reduced = Mat.raw_ref(self.valuelist, True)  # get to REF to multiply entries on leading diagonal

        if not self.issquare():  # det(a) = 0 if a is not square
            return 0

        pivots = 1
        zerotest = 0

        for row in reduced:
            for entry in row:
                if entry != 0:
                    zerotest += 1  # check that there is a pivot in each row (so that det != 0)
                    pivots *= entry  # multiply entries on leading diagonal to get determinant
                    break

        if zerotest == len(reduced):  # meaning that there is a pivot in each row
            det = (-1) ** swaps * pivots  # make sure sign of determinant is correct

            if abs(det - round(det, 1)) < 0.00000001:  # round for aesthetics
                det = round(det, 1)
            elif abs(det - round(det)) < 0.00000001:
                det = round(det)
        else:
            det = 0  # if not invertible, determinant = 0

        return det

    def inverse(self):  # caluculate inverse of matrix
        augmented = deepcopy(self.valuelist)  # avoid altering self.valuelist

        if self.det() == 0:  # if det(a) = 0, a is not invertible
            raise TypeError('Matrix is not invertible.')

        dim = len(augmented)

        for index, row in enumerate(augmented):  # create augmented matrix with identity matrix on the right
            row.extend([1 if i == index else 0 for i in range(dim)])

        rrefaug = Mat.raw_rref(augmented)  # get to augmented matrix to RREF

        inversemat = []

        for row in rrefaug:
            inversemat.append(row[dim:])  # inverse is right half of reduced augmented matrix

        return Mat(inversemat)

    def issquare(self):  # for use in self.det()
        if len(self.valuelist) == len(self.valuelist[0]):
            return True  # only True if number of columns = number of rows
        else:
            return False

    def samedimensions(self, other):  # for use in self.__add__ and __sub__
        if len(self.valuelist) != len(other.valuelist):
            return False

        for row, otherrow in zip(self.valuelist, other.valuelist):
            if len(row) != len(otherrow):
                return False  # makes sure that both matrices have same dimensions

        return True

    @classmethod
    def from_input(cls):  # for convenience when inputting entries for a large matrix
        counter = 0

        while True:  # repeat until user enters acceptable values
            try:
                if counter == 0:  # ask for amount of rows
                    m = int(input('Enter number of matrix rows: '))

                    if m < 1:  # rows must be > 0
                        raise IndexError

                    counter = 1
                # ask for amount of columns
                n = int(input('Enter number of matrix columns: '))

                if n < 1:  # columns must be > 0
                    raise IndexError

                break

            except ValueError:  # if user enters non-numeric entries
                print('Please enter an integer.')
            except IndexError:  # if user enters matrix with a dimension of 0
                print('Your number must be greater than 1.')

        fulllist = []

        for i in range(m):  # go through each row
            while True:  # repeat until user enters acceptable values
                try:
                    rawrow = input(f'Enter row {i + 1} numbers, separated by spaces: ')

                    row = []

                    for entry in rawrow.split():  # split input string into list of numbers
                        if '.' in entry:
                            row.append(float(entry))
                        else:
                            row.append(int(entry))

                    if len(row) != n:  # if user enters different amount of numbers than columns
                        raise IndexError

                    fulllist.append(row)

                    break

                except ValueError:  # if user enters non-numeric entries
                    print('Please only enter numbers.')
                except IndexError:  # if user tries to input non-rectangular matrix
                    print(f'Please enter {n} values per row.')

        object = cls(fulllist)  # create Mat object from list

        return object


# if run as a script, give a few examples of functionality
if __name__ == '__main__':
    a = Mat([[3, -7, 2], [1.4, 0, -2], [9.51, 5, -0.8]])  # invertible example
    print(a)  # show __str__ method
    print()

    b = a.det()  # show determinant
    print(b)
    print()

    c = a.rref()  # show reduced row-echelon form
    print(c)
    print()

    d = a.inverse()  # show inverse
    print(a * d)  # check inverse
    print()
    print(d * a)
    print()

    e = Mat.from_input()  # try with a nonsquare matrix, maybe 3 x 4
    print(e)
    print()
    print(e.transpose())  # show transpose
    print()
    print(e.rref())  # show RREF again
