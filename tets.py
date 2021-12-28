import inspect


class B:
    a=5
    b='dsadsa'
    fk_bla = 'd'

    def get_l(self):
        return None



if __name__ == '__main__':
    class_attributes = B().__class__.__dict__
    foreign_keys_attributs = [attribute[0] for attribute in inspect.getmembers(B) if 'fk' in attribute[0]]

    print(foreign_keys_attributs)
