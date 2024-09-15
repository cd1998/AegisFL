from seal import *
import numpy as np

parms = EncryptionParameters(scheme_type.ckks)
poly_modulus_degree = 8192
scale = 2 ** 40
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))
parms.set_encoding_method(encoding_method.pm)
context = SEALContext(parms)
keygen = KeyGenerator(context)
secret_key = keygen.secret_key()
public_key = keygen.create_public_key()
relin_keys = keygen.create_relin_keys()

encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)
encoder = CKKSEncoder(context)

def encrypt(a,model_len):
    mod = np.mod(model_len, poly_modulus_degree)
    data1 = np.concatenate([a, np.zeros(poly_modulus_degree - mod)])
    data2 = np.concatenate([a, np.zeros(poly_modulus_degree - mod)])

    num_batches = len(data1) // poly_modulus_degree
    encrypt_result_1 = []
    encrypt_result_2 = []
    for i in range(num_batches):
        start_idx = i * poly_modulus_degree
        end_idx = (i + 1) * poly_modulus_degree
        batch_1 = data1[start_idx:end_idx]
        batch_2 = data2[start_idx:end_idx]
        reversed_part = -batch_2[-1:0:-1]
        result_array = np.concatenate(([batch_2[0]], reversed_part))
        batch_2 = result_array

        plain1 = encoder.encode(batch_1, scale)
        plain2 = encoder.encode(batch_2, scale)
        cipher1 = encryptor.encrypt(plain1)
        cipher2 = encryptor.encrypt(plain2)

        encrypt_result_1.append(cipher1)
        encrypt_result_2.append(cipher2)
    return encrypt_result_1, encrypt_result_2

def sub(a,b):
    result = []
    for i in range(len(a)):
        c = evaluator.sub(a[i],b[i])
        result.append(c)
    return result

def add(a,b):
    result = []
    for i in range(len(a)):
        c = evaluator.add(a[i],b[i])
        result.append(c)
    return result


def decrypt(a):
    decrypt = []
    for i in range(len(a)):
        dec = decryptor.decrypt(a[i])
        pla = encoder.decode(dec)
        decrypt.append(pla)
    return decrypt

def weight_sum(cipher, weight):
    len1 = len(cipher)
    len2 = len(cipher[1])
    weight_poly = []
    for i in range(len1):
        weight_numpy = np.zeros(8192)
        weight_numpy[0] = weight[i]
        poly = encoder.encode(weight_numpy, scale)
        print('eeeeeeeeeee',weight_numpy)
        weight_poly.append(poly)
    result = []

    for i in range(len2):
        a = evaluator.multiply_plain(cipher[0][i], weight_poly[0])
        for j in range(1,len1):
            b = evaluator.multiply_plain(cipher[j][i], weight_poly[j])
            evaluator.add_inplace(a, b)
        evaluator.rescale_to_next_inplace(a)
        a.scale(2**40)
        result.append(a)
    return result


# def rescale(cipher):
#     length = len(cipher)
#     result = []
#     for i in range(length):
#         evaluator.rescale_to_next_inplace(cipher[i])
#         result.append(cipher[i])
#     return result

def modswitch(cipher,id):
    length = len(cipher)
    result = []
    for i in range(length):
        a = evaluator.mod_switch_to(cipher[i], id)
        result.append(a)
    return result

