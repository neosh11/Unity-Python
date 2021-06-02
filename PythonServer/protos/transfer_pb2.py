# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: transfer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='transfer.proto',
  package='transfer',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0etransfer.proto\x12\x08transfer\"\x8b\x01\n\x08Transfer\x12\x37\n\x0etypeOfFunction\x18\x01 \x01(\x0e\x32\x1f.transfer.Transfer.TypeFunction\x12\x10\n\x08\x64\x63mFiles\x18\x02 \x03(\x0c\"4\n\x0cTypeFunction\x12\x08\n\x04NONE\x10\x00\x12\x0f\n\x0b\x42\x45\x44_REMOVAL\x10\x01\x12\t\n\x05MAGIC\x10\x02\x62\x06proto3'
)



_TRANSFER_TYPEFUNCTION = _descriptor.EnumDescriptor(
  name='TypeFunction',
  full_name='transfer.Transfer.TypeFunction',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BED_REMOVAL', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='MAGIC', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=116,
  serialized_end=168,
)
_sym_db.RegisterEnumDescriptor(_TRANSFER_TYPEFUNCTION)


_TRANSFER = _descriptor.Descriptor(
  name='Transfer',
  full_name='transfer.Transfer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='typeOfFunction', full_name='transfer.Transfer.typeOfFunction', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dcmFiles', full_name='transfer.Transfer.dcmFiles', index=1,
      number=2, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _TRANSFER_TYPEFUNCTION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=168,
)

_TRANSFER.fields_by_name['typeOfFunction'].enum_type = _TRANSFER_TYPEFUNCTION
_TRANSFER_TYPEFUNCTION.containing_type = _TRANSFER
DESCRIPTOR.message_types_by_name['Transfer'] = _TRANSFER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Transfer = _reflection.GeneratedProtocolMessageType('Transfer', (_message.Message,), {
  'DESCRIPTOR' : _TRANSFER,
  '__module__' : 'transfer_pb2'
  # @@protoc_insertion_point(class_scope:transfer.Transfer)
  })
_sym_db.RegisterMessage(Transfer)


# @@protoc_insertion_point(module_scope)