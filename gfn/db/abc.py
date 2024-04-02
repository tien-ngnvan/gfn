from abc import ABC, abstractmethod


class FaceDatabaseInterface(ABC):

  @abstractmethod
  def create_person(self, person_id):
    return NotImplemented
  
  def delete_person(self, person_id):
    return
  
  @abstractmethod
  def insert_face(self, person_id, face_id, face_emb):
    return NotImplemented
  
  @abstractmethod
  def delete_face(self, person_id, face_id):
    return 
  
  @abstractmethod
  def check_face(self, person_id, face_emb):
    return
